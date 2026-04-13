import argparse
import json
import os
import pickle
import threading

import torch
from dataloader import *

parser = argparse.ArgumentParser('')
parser.add_argument('--dataset_name', type=str, default="video-dataset", metavar='DN',
                    help='dataset name for finding annotation files')
parser.add_argument('--trace_path', type=str, default="/pickle/file/format", metavar='TP',
                    help='path to a file which is a list containing paths of extracted traces')
parser.add_argument('--clip_score_dir', type=str, default="/path/to/dir/clip_filtered_scores/", metavar='CSD',
                    help='path to directory which contains the clip scores')
parser.add_argument('--output_path', type=str, default="/path/to/output/file", metavar='OVD',
                    help='path to output json file containing list of valid traces')

parser.add_argument('--min_score', type=float, default=0.25, metavar='MS',
                    help='number of frames to use per video')
parser.add_argument('--split_idx', type=int, default=0, metavar='SI',
                    help='index for splitting entire dataset over multiple GPUs')
parser.add_argument('--num_samples_per_segment', type=int, default=10400145, metavar='NS',
                    help='specify number of segments per GPU')
parser.add_argument('--num_workers', type=int, default=8, metavar='NW',
                    help='number of worker processes')
parser.add_argument('--batch_size', type=int, default=128, metavar='BS',
                    help='batch size')
parser.add_argument('--thread_num', type=int, default=72, metavar='TN',
                    help='number of threads')

valid_traces = []
full_set_traces = []

def main():
    global args
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output_path
    all_traces = list(pickle.load(open(args.trace_path, 'rb'))) # This should be a list that contains the paths to all extracted traces

    print('all_traces: ', len(all_traces))
    print('')

    lock = threading.Lock()

    # Function that writes to the global set
    def add_to_set(tid, split_traces):
        print('split %s: ' % tid, len(split_traces))
        print('')
        for idx, trace in enumerate(split_traces):
            if tid == 0 and idx % 1000 == 0:
                print(idx)

            trace_path = trace[0] + '/' + trace[-1]
            score_path = os.path.join(args.clip_score_dir, trace[0], '%s.pth' % trace[-1])

            try:
                trace_score = torch.load(score_path, map_location='cpu').max()
                if trace_score >= args.min_score:
                    global valid_traces
                    with lock:  # Ensure that only one thread can modify the set at a time
                        valid_traces.append(trace_path)

                global full_set_traces
                with lock:
                    full_set_traces.append(trace_path)
            except:
                continue


    # Create threads
    per_process_video_num = len(all_traces) // args.thread_num

    threads = []
    for i in range(args.thread_num):
        if i == args.thread_num - 1:
            sub_files = all_traces[i * per_process_video_num :]
        else:
            sub_files = all_traces[i * per_process_video_num : (i + 1) * per_process_video_num]

        t = threading.Thread(target=add_to_set, args=(i, sub_files,))
        threads.append(t)
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    json.dump(valid_traces, open(output_path, 'w'))
    print('valid_traces: ', len(valid_traces))
    print('full_set_traces: ', len(full_set_traces))
    
if __name__ == "__main__":
    main()