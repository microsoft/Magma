import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def som_prompting(image, pos_traces, neg_traces, draw_som_positive=False, draw_som_negative=False):
    """
    draw marks on the image
    """
    image_size = image.size
    draw = ImageDraw.Draw(image)

    def get_text_size(text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    def expand_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return [x1-4, y1-4, x2+4, y2+4]
    
    def draw_marks(draw, points, text_size, id, font_size):
        txt = str(id)
        draw.ellipse(((points[0]-max(text_size)//2-1, points[1]-max(text_size)//2-1, points[0]+max(text_size)//2+1, points[1]+max(text_size)//2+1)), fill='red')
        draw.text((points[0]-text_size[0] // 2, points[1]-text_size[1] // 2-3), txt, fill='white', font=font_size)
        
    fontsize = 1
    font = ImageFont.truetype("data/utils/arial.ttf", fontsize)
    txt = "55"    
    while min(get_text_size(txt, image, font)) < 0.03*image_size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("data/utils/arial.ttf", fontsize)

    text_size_2digits = get_text_size('55', image, font)
    text_size_1digit = get_text_size('5', image, font)
    text_size = {
        1: text_size_1digit,
        2: text_size_2digits
    }

    # draw the starting point of positive traces on image
    num_pos = pos_traces.shape[2]
    pos_idx = torch.arange(num_pos)
    pos_traces_to_mark = pos_traces

    # random sample at most 10 negative traces
    num_neg = neg_traces.shape[2]
    neg_idx = torch.arange(num_neg)
    neg_traces_to_mark = neg_traces

    num_traces_total = pos_traces_to_mark.shape[2] + neg_traces_to_mark.shape[2]
    # shuffle the indices
    all_idx = torch.randperm(num_traces_total)

    pos_mark_ids = []; neg_mark_ids = []
    pos_traces_som = {}
    for i in range(pos_traces_to_mark.shape[2]):
        pos = pos_traces_to_mark[:,:,i]
        mark_id = all_idx[i].item()
        text_size = get_text_size(str(mark_id+1), image, font)
        if draw_som_positive:
            draw_marks(draw, pos[0][0], text_size, mark_id+1, font)
        pos_traces_som[mark_id+1] = pos
        pos_mark_ids.append(mark_id+1)
    
    neg_traces_som = {}
    for i in range(neg_traces_to_mark.shape[2]):
        neg = neg_traces_to_mark[:,:,i]
        mark_id = all_idx[pos_traces_to_mark.shape[2]+i].item()
        text_size = get_text_size(str(mark_id+1), image, font)
        if draw_som_negative:
            draw_marks(draw, neg[0][0], text_size, mark_id+1, font)
        neg_traces_som[mark_id+1] = neg
        neg_mark_ids.append(mark_id+1)

    return image, pos_traces_som, neg_traces_som, pos_mark_ids, neg_mark_ids, all_idx

def som_prompting_with_priors(image, pos_traces_som, neg_traces_som, pos_mark_ids, neg_mark_ids, all_idx, step_offset=1, draw_som_positive=False, draw_som_negative=False):
    """
    draw marks on the image
    """
    image_size = image.size
    draw = ImageDraw.Draw(image)

    def get_text_size(text, image, font):
        im = Image.new('RGB', (image.width, image.height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return width, height
    
    def expand_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return [x1-4, y1-4, x2+4, y2+4]
    
    def draw_marks(draw, points, text_size, id, font_size):
        txt = str(id)
        draw.ellipse(((points[0]-max(text_size)//2-1, points[1]-max(text_size)//2-1, points[0]+max(text_size)//2+1, points[1]+max(text_size)//2+1)), fill='red')
        draw.text((points[0]-text_size[0] // 2, points[1]-text_size[1] // 2-3), txt, fill='white', font=font_size)
        
    fontsize = 1
    font = ImageFont.truetype("data/utils/arial.ttf", fontsize)
    txt = "55"    
    while min(get_text_size(txt, image, font)) < 0.02*image_size[0]:
        # iterate until the text size is just larger than the criteria
        fontsize += 1
        font = ImageFont.truetype("data/utils/arial.ttf", fontsize)

    text_size_2digits = get_text_size('55', image, font)
    text_size_1digit = get_text_size('5', image, font)
    text_size = {
        1: text_size_1digit,
        2: text_size_2digits
    }

    for key, val in pos_traces_som.items():
        mark_id = key
        pos = val[:,step_offset if step_offset < val.shape[1] else -1]
        text_size = get_text_size(str(mark_id), image, font)
        if draw_som_positive:
            draw_marks(draw, pos[0], text_size, mark_id, font)
    
    for key, val in neg_traces_som.items():
        mark_id = key
        neg = val[:,step_offset if step_offset < val.shape[1] else -1]
        text_size = get_text_size(str(mark_id), image, font)
        if draw_som_negative:
            draw_marks(draw, neg[0], text_size, mark_id, font)

    return image

def tom_prompting(trace, image, pos_traces, neg_traces, draw_som_positive=False, draw_som_negative=False):
    """
    draw trace-of-marks on the image
    """
    image_size = image.size
    # draw traces for all points
    # get all traces
    tracks = torch.cat([pos_traces, neg_traces], dim=2).cpu().numpy()
    _, T, N, _ = tracks.shape    
    vector_colors = np.zeros((T, N, 3))
    if trace.visualizer.mode == "rainbow":
        y_min, y_max = (
            tracks[0, 0, :, 1].min(),
            tracks[0, 0, :, 1].max(),
        )
        norm = plt.Normalize(y_min, y_max)
        for n in range(N):
            color = trace.visualizer.color_map(norm(tracks[0, 0, n, 1]))
            color = np.array(color[:3])[None] * 255
            vector_colors[:, n] = np.repeat(color, T, axis=0)
    else:
        # color changes with time
        for t in range(T):
            color = np.array(trace.visualizer.color_map(t / T)[:3])[None] * 255
            vector_colors[t] = np.repeat(color, N, axis=0)

    # PIL to numpy
    image = np.array(image).astype(np.uint8)
    # unsqueeze image to 4D
    curr_tracks = tracks[0]
    curr_colors = vector_colors
    image = trace.visualizer._draw_pred_tracks(image, curr_tracks, curr_colors)
    image = Image.fromarray(image)
    return image