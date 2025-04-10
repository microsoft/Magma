import os
import json
import cv2
import csv
import io
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

from cotracker.utils.visualizer import Visualizer
from data.utils.visual_trace import visual_trace
from data.utils.som_tom import som_prompting, tom_prompting

device = 'cuda:1'
grid_size = 15
cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
visual_trace_folder = "./tools/som_tom/videos"
vis = Visualizer(save_dir=visual_trace_folder, pad_value=0, linewidth=3, tracks_leave_trace=-1)
trace = visual_trace(linewidth=3)

def som_tom(video, pred_tracks, pred_visibility, item={}, epsilon=2):
    # only keep points that are visible at at least half steps
    valid_idx = pred_visibility[0].sum(0) > 0.5*pred_tracks.shape[1]
    pred_tracks = pred_tracks[:, :, valid_idx]
    pred_visibility = pred_visibility[:, :, valid_idx]

    # Alg2 L2-4: Remove camera motion
    # calculate the trajectory lenght for pred_tracks
    trace_lengths = trace.visual_trace_length(pred_tracks, pred_visibility, (1, 1)).squeeze(0)
    # if 80% of the pred_tracks_length is larger than 2, then there is camera motion
    camera_motion = (trace_lengths > 0.5).sum() > 0.8*trace_lengths.size(0)
    start_pos = pred_tracks[:, 0][0]
    reference_pts_np = start_pos.cpu().numpy().reshape(-1, 2)

    if camera_motion:
        # remove camera motion using homography transformation
        try:
            future_pts_transformed = []
            for k in range(1, pred_tracks.shape[1]):
                future_pts = pred_tracks[:, k][0]
                future_pts_np = future_pts.cpu().numpy().reshape(-1, 2)
                try:
                    (H, status) = cv2.findHomography(future_pts_np, reference_pts_np, cv2.RANSAC, 4.0)
                except Exception as e:
                    continue
                future_pts_np_transformed = cv2.perspectiveTransform(future_pts_np.reshape(1, -1, 2), H).reshape(-1, 2)                                
                future_pts_transformed_k = torch.tensor(future_pts_np_transformed, dtype=torch.float32)
                future_pts_transformed.append(future_pts_transformed_k)            
            pred_tracks = torch.stack([start_pos] + future_pts_transformed, dim=0).unsqueeze(0)           
        except Exception as e:
            pass
    
    # Alg2 L5: Find the positive traces and negative traces
    pos_tracks = pred_tracks[:, :, trace_lengths > epsilon]
    pos_visibility = pred_visibility[:, :, trace_lengths > epsilon]
    neg_tracks = pred_tracks[:, :, trace_lengths <= epsilon]
    neg_visibility = pred_visibility[:, :, trace_lengths <= epsilon]

    # Alg2 L6-7: Clustering for positive and negative traces
    num_clusters_pos = torch.randint(2, 6, (1,)).item()
    pos_sampled_ids = trace.cluster_traces_kmeans(pos_tracks, n_clusters=num_clusters_pos, positive=True)
    pos_tracks = pos_tracks[:, :, pos_sampled_ids.bool()]
    pos_visibility = pos_visibility[:, :, pos_sampled_ids.bool()]

    # clustering for negative traces
    num_clusters_neg = torch.randint(6, 15, (1,)).item()
    neg_sampled_ids = trace.cluster_traces_kmeans(neg_tracks, n_clusters=num_clusters_neg)
    neg_tracks = neg_tracks[:, :, neg_sampled_ids.bool()]

    image = video[0][0].numpy().transpose(1, 2, 0).astype(np.uint8)
    image = Image.fromarray(image).convert("RGB")

    # Alg2 L8: Apply som on the first frame
    image, pos_traces_to_mark, neg_traces_to_mark, pos_mark_ids, neg_mark_ids, all_idx = \
        som_prompting(image, pos_tracks, neg_tracks, draw_som_positive=True, draw_som_negative=True)

    # visualize the traces
    images = [image] * pos_tracks.shape[1]
    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    _ = vis.visualize(video, pos_tracks, pos_visibility, filename=f"som_tom")

video_path = "assets/videos/clippedUE_h264.mp4"
# load video
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
# get number of frames in cap
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
images = []
frame_count = 0
max_frames = 120  # Or another suitable number

while True:
    ret, frame = cap.read()
    # if reach stop frame then break
    if not ret or frame_count >= max_frames:
        break
        
    # Convert BGR to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    images.append(frame_rgb)
    frame_count += 1

cap.release()

# Add error checking before processing
if len(images) == 0:
    print("Error: No frames were read from the video")
    # Handle the error (e.g., exit or use default image)
else:
    images = [Image.fromarray(img) for img in images]
    # resize images to height=512
    images = [img.resize((int(img.width * 512 / img.height), 512)) for img in images]

    video = torch.stack([torchvision.transforms.ToTensor()(img) for img in images])[None].float()*255    
    video = video.to(device)

    # Alg2 L1: Extract visual trace
    pred_tracks, pred_visibility = cotracker(video, grid_size=grid_size) # B T N 2,  B T N 1
    _ = vis.visualize(
        video.cpu(),
        pred_tracks,
        pred_visibility,
        query_frame=0,
        filename='orig_trace',
    )        

    som_tom(video.cpu(), pred_tracks, pred_visibility)