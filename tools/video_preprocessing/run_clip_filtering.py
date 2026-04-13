import argparse
import concurrent.futures
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import clip
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage

parser = argparse.ArgumentParser('')
parser.add_argument('--dataset_name', type=str, default="video-dataset", metavar='DN',
                    help='dataset name for finding annotation files')
parser.add_argument('--trace_path', type=str, default="/path/to/dir/for/extracted/traces", metavar='TP',
                    help='path to directory with extracted traces')
parser.add_argument('--ann_path', type=str, default="/path/to/processed/language/annotations", metavar='AP',
                    help='path to language annotations. Process this into a dictionary in any format you prefer. We save this as a pickle file.')
parser.add_argument('--input_video_dir', type=str, default="/path/to/video/dir", metavar='IVD',
                    help='path to input video directory')
parser.add_argument('--output_dir', type=str, default="/path/to/clip_filtered_scores/", metavar='OD',
                    help='path to store clip filtered scores')

parser.add_argument('--filtered', type=bool, default=True, metavar='FIL',
                    help='boolean flag to specify if a filtered list is used')
parser.add_argument('--num_frames', type=int, default=4, metavar='NF',
                    help='number of frames to use per video')
parser.add_argument('--max_segment_frames', type=int, default=16, metavar='MSF',
                    help='maximum number of frames per segment')
parser.add_argument('--num_workers', type=int, default=8, metavar='NW',
                    help='number of worker processes')
parser.add_argument('--batch_size', type=int, default=2, metavar='BS',
                    help='batch size')

@dataclass
class DataCollator(object):
    """Collate examples."""

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        frames = []
        text = []
        num_frames = []
        segment = []
        for instance in instances:
            frames.append(instance['frames'])
            text += instance['text']
            num_frames.append(instance['num_frames'])
            segment.append(instance['segment'])

        frames = torch.stack(frames)

        return {'frames': frames , 'text': text, 'num_frames': num_frames, 'segment': segment}

class VideoDataset(Dataset):
    def __init__(self, args, img_processor):
        self.args = args
        self.dataset_name = args.dataset_name
        self.video_dir = args.input_video_dir
        self.max_segment_frames = args.max_segment_frames
        self.num_frames = args.num_frames
        self.split_idx = args.split_idx
        self.num_samples_per_segment = args.num_samples_per_segment
        self.filtered = args.filtered
        self.img_processor = img_processor
        self.to_pil = ToPILImage()
        self.vid2anns = pickle.load(open(args.ann_path, 'rb')) # a nested map from video to language annotations. For example, the keys at the first level will be the video and those at the second level are segments (denoted by start and end times). The values are the language annotations.
        self.all_traces = pickle.load(open(args.trace_path, 'rb'))

    def __len__(self):
        return len(self.all_traces)

    def __getitem__(self, idx):
        curr_segment = self.all_traces[idx]
        vid, trace_id = curr_segment
        tmp = trace_id.split('_trace_')
        clip = tmp[0]
        times = tmp[-1].split('_')
        start = int(times[0])
        end = int(times[1])

        intervals = trace_id.split('___')
        time_start = float(intervals[0].split('_')[-1])
        time_end = float(intervals[1].split('_')[-1])

        try:
            video_path = os.path.join(self.video_dir, vid, '%s.mp4' % clip)
            frames, _, _ = torchvision.io.read_video(video_path)
            frames = frames.permute(0, 3, 1, 2)

            selected_frames = []
            selected_indices = torch.linspace(start, end-1, min(self.num_frames, end-start+1))
            for i in selected_indices:
                selected_frames.append(self.img_processor(self.to_pil(frames[int(i)])))
            selected_frames = torch.stack(selected_frames)

            trace_num_frames = len(selected_frames)
            if len(selected_frames) < self.num_frames:
                last = selected_frames[-1].unsqueeze(0).repeat(self.num_frames-len(selected_frames), 1, 1, 1)
                selected_frames = torch.cat((selected_frames, last), dim=0)

            time_start, time_end, _ =  clip.split('___')
            time_start = float(time_start.split('start_')[-1])
            time_end = float(time_end.split('end_')[-1])
            if self.dataset_name == 'howto100m':
                text = self.vid2anns[vid][(time_start, time_end)]
            else:
                narr = int(clip.split('narr_')[-1])
                text = self.vid2anns[vid][(time_start, time_end, narr)]
        except:
            selected_frames = torch.zeros((self.num_frames, 3, 336, 336))
            text = 'none'
            trace_num_frames = 0

        return {'frames': selected_frames, 'text': [text], 'num_frames': trace_num_frames, 'segment': [curr_segment]}

def save_tensor(tensor, file_path):
    torch.save(tensor, file_path)

def main():
    global args
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model, preprocess = clip.load("ViT-L/14@336px", device='cuda')
    model.eval()

    dataset = VideoDataset(args, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=DataCollator()
    )

    for idx, sample in enumerate(dataloader):

        print(idx)

        frames = sample['frames']
        text = sample['text']
        num_frames = sample['num_frames']
        segment = sample['segment']

        batch_size = len(frames)
        frames = frames.view(-1, frames.size(-3), frames.size(-2), frames.size(-1)).to('cuda')
        text_tokens = clip.tokenize(text, truncate=True).to('cuda')
        text_tokens = text_tokens.unsqueeze(1).repeat(1, args.num_frames, 1)
        text_tokens = text_tokens.view(-1, text_tokens.size(-1))

        with torch.no_grad():
            frame_features = model.encode_image(frames)
            text_features = model.encode_text(text_tokens)

            # normalized features
            frame_features = frame_features / frame_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            scores = torch.sum(frame_features * text_features, dim=-1)
            scores = scores.view(batch_size, -1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for sample_idx in range(len(scores)):
                curr_num = num_frames[sample_idx]

                if curr_num == 0:
                    continue

                curr_scores = scores[sample_idx][:curr_num]
                curr_vid, curr_trace = segment[sample_idx][0] # each video (curr_vid) is split into segments that are further split into windows of 16 frames (curr_trace). E.g, Video1, segment_4_frames_16_32
                output_vid_path = os.path.join(args.output_dir, curr_vid)
                curr_output_path = os.path.join(output_vid_path, '%s.pth' % curr_trace)

                if not os.path.exists(output_vid_path):
                    os.mkdir(output_vid_path)

                futures.append(executor.submit(save_tensor, curr_scores.cpu(), curr_output_path))

            concurrent.futures.wait(futures)

if __name__ == "__main__":
    main()