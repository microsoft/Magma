import argparse
import json
import multiprocessing as mp
import os

import cv2
import imageio
import numpy as np
import torch
from scenedetect import ContentDetector, detect

parser = argparse.ArgumentParser('')
parser.add_argument('--ann_path', type=str, default="/path/to/json/file", metavar='AP',
                    help='path to json file that contains video and language annotations. See lines 169 - 172 for more detail.')
parser.add_argument('--video_dir', type=str, default="/path/to/video/directory", metavar='VD',
                    help='path to video dir')
parser.add_argument('--temp_video_segment_dir', type=str, default="./temp_video_segments", metavar='TD',
                    help='temporary directory to store split video segments')
parser.add_argument('--output_segment_dir', type=str, default="./detected_video_segments", metavar='OD',
                    help='path to store the final detected video segments')
parser.add_argument('--target_fps', type=int, default=None, metavar='TFPS',
                    help='FPS to sample frames')
parser.add_argument('--thread_num', type=int, default=8, metavar='TN',
                    help='number of threads')

def extract_video_frames_and_metadata(video_path, target_fps=1):
    '''
    Extracts video frames at 1 fps by default
    '''
    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)

    round_vid_fps = round(vid_fps)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = num_frames / round_vid_fps

    if target_fps is not None:
        hop = round(vid_fps / target_fps)

    all_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if target_fps is not None:
            if frame_idx % hop == 0:
                all_frames.append(frame)
        else:
            all_frames.append(frame)

        frame_idx += 1

    cap.release()
    return vid_fps, num_frames, duration, all_frames

def write_video(video, output_path, write_fps):
    wide_list = list(video.unbind(1))
    wide_list = [wide[0].permute(1, 2, 0).cpu().numpy() for wide in wide_list]
    video_writer = imageio.get_writer(output_path, fps=write_fps)
    for frame in wide_list[2:-1]:
        video_writer.append_data(frame)
    video_writer.close()
    return

def extract_num_frames(video_path):
    '''
    Extracts video frames at 1 fps
    '''
    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    round_vid_fps = round(vid_fps)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    return num_frames, vid_fps, cap

def process_single_vid(vid, vid_anns, min_part_duration=3):
    vid_path = os.path.join(args.video_dir, '%s.mp4' % vid)
    num_frames, vid_fps, cap = extract_num_frames(vid_path)
    vid_fps = int(vid_fps)

    start = vid_anns['start']
    end = vid_anns['end']
    text = vid_anns['text']

    for idx, curr_start in enumerate(start):
        curr_end = end[idx]
        curr_text = text[idx]

        full_segment_path = os.path.join(args.temp_video_segment_dir, '%s___start_%s___end_%s.mp4' % (vid, curr_start, curr_end))

        actual_start = curr_start * vid_fps
        actual_end = curr_end * vid_fps
        actual_num_frames = int(actual_end - actual_start + 1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)
        all_frames = []
        for frame_idx in range(actual_num_frames):
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame)

        all_frames = np.stack(all_frames)
        all_frames = torch.from_numpy(all_frames)
        all_frames = all_frames.permute(0, 3, 1, 2)[None].byte()

        write_video(all_frames, full_segment_path, vid_fps)
        scene_list = detect(full_segment_path, ContentDetector())

        output_dir = os.path.join(args.output_segment_dir, vid)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if len(scene_list) == 0:
            split_segment_path = os.path.join(output_dir, 'start_%s___end_%s___part_%s.mp4' % (curr_start, curr_end, 0))
            if os.path.exists(split_segment_path):
                continue
            write_video(all_frames, split_segment_path, vid_fps)
        else:
            for part_idx, scene in enumerate(scene_list):
                first = scene[0].get_frames()
                second = scene[1].get_frames()

                split_segment_path = os.path.join(output_dir, 'start_%s___end_%s___part_%s.mp4' % (curr_start, curr_end, part_idx))
                if os.path.exists(split_segment_path):
                    continue

                write_video(all_frames[:, first:second+1], split_segment_path, vid_fps)

        cmd = 'rm %s' % full_segment_path
        os.system(cmd)

    return

def sub_processor(pid, files, data):
    print(pid, ' : ', len(files))
    for curr_vid in files[:]:
        try:
            vid_anns = data[curr_vid]
            process_single_vid(curr_vid, vid_anns)
        except:
            continue

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.output_segment_dir):
        os.mkdir(args.output_segment_dir)

    # This assumes that we have language annotations that are stored in a nested dictionary
    # The keys at the first level are the video names or ids
    # The values are dictionaries that contain the start and end times as well as the text
    # This format can be easily modified to suit different datasets.
    data = json.load(open(args.ann_path)) 
                                             
    video2anns = {}
    for idx, vid in enumerate(data):
        if idx % 100 == 0:
            print(idx)
        
        curr = data[vid]
        start = curr['start']
        end = curr['end']
        text = curr['text']

        if vid not in video2anns:
            video2anns[vid] = {}
        video2anns[vid][start] = narr

    #json.dump(video2anns, open('/path/to/vid_to_anns.json', 'w'))
    all_vids = list(video2anns.keys())

    print('all_vids: ', len(all_vids))
    print('')

    processes = []
    video_num = len(all_vids)
    per_process_video_num = video_num // args.thread_num

    for i in range(args.thread_num):
        if i == args.thread_num - 1:
            sub_files = all_vids[i * per_process_video_num :]
        else:
            sub_files = all_vids[i * per_process_video_num : (i + 1) * per_process_video_num]
        p = mp.Process(target=sub_processor, args=(i, sub_files, video2anns))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()