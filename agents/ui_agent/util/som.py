from PIL import Image

device = 'cuda'

import networkx as nx
import numpy as np
from PIL import ImageDraw, ImageFont

# import cv2

font_path = "agents/ui_agent/util/arial.ttf"
class MarkHelper:
    def __init__(self):    
        self.markSize_dict = {}
        self.font_dict = {}
        self.min_font_size = 20 # 1 in v1
        self.max_font_size = 30
        self.max_font_proportion = 0.04 # 0.032 in v1

    def __get_markSize(self, text, image_height, image_width, font):
        im = Image.new('RGB', (image_width, image_height))
        draw = ImageDraw.Draw(im)
        _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
        return height, width

    def _setup_new_font(self, image_height, image_width):
        key = f"{image_height}_{image_width}"
        # print(f"Setting up new font for image size: {key}")
        
        # setup the font
        fontsize = self.min_font_size
        font = ImageFont.truetype(font_path, fontsize)
        # font = ImageFont.load_default(size=fontsize)
        while min(self.__get_markSize("555", image_height, image_width, font)) < min(self.max_font_size, self.max_font_proportion * min(image_height, image_width)):
            # iterate until the text size is just larger than the criteria
            fontsize += 1
            font = ImageFont.truetype(font_path, fontsize)
            # font = ImageFont.load_default(size=fontsize)
        self.font_dict[key] = font

        # setup the markSize dict
        markSize_3digits = self.__get_markSize('555', image_height, image_width, font)
        markSize_2digits = self.__get_markSize('55', image_height, image_width, font)
        markSize_1digit = self.__get_markSize('5', image_height, image_width, font)
        self.markSize_dict[key] = {
            1: markSize_1digit,
            2: markSize_2digits,
            3: markSize_3digits
        }

    def get_font(self, image_height, image_width):
        key = f"{image_height}_{image_width}"
        if key not in self.font_dict:
            self._setup_new_font(image_height, image_width)
        return self.font_dict[key]
        
    def get_mark_size(self, text_str, image_height, image_width):
        """Get the font size for the given image dimensions."""
        key = f"{image_height}_{image_width}"
        if key not in self.markSize_dict:
            self._setup_new_font(image_height, image_width)

        largest_size = self.markSize_dict[key].get(3, None)
        text_h, text_w = self.markSize_dict[key].get(len(text_str), largest_size) # default to the largest size if the text is too long
        return text_h, text_w

def __calculate_iou(box1, box2, return_area=False):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    :param box1: Tuple of (y, x, h, w) for the first bounding box
    :param box2: Tuple of (y, x, h, w) for the second bounding box
    :return: IoU value
    """
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2

    # Calculate the intersection area
    y_min = max(y1, y2)
    x_min = max(x1, x2)
    y_max = min(y1 + h1, y2 + h2)
    x_max = min(x1 + w1, x2 + w2)

    intersection_area = max(0, y_max - y_min) * max(0, x_max - x_min)

    # Compute the area of both bounding boxes
    box1_area = h1 * w1
    box2_area = h2 * w2

    # Calculate the IoU
    # iou = intersection_area / box1_area + box2_area - intersection_area
    iou = intersection_area / (min(box1_area, box2_area) + 0.0001)

    if return_area:
        return iou, intersection_area
    return iou

def __calculate_nearest_corner_distance(box1, box2):
    """Calculate the distance between the nearest edge or corner of two bounding boxes."""
    y1, x1, h1, w1 = box1
    y2, x2, h2, w2 = box2
    corners1 = np.array([
        [y1, x1],
        [y1, x1 + w1],
        [y1 + h1, x1],
        [y1 + h1, x1 + w1]
    ])
    corners2 = np.array([
        [y2, x2],
        [y2, x2 + w2],
        [y2 + h2, x2],
        [y2 + h2, x2 + w2]
    ])
    # Calculate pairwise distances between corners
    distances = np.linalg.norm(corners1[:, np.newaxis] - corners2, axis=2)

    # Find the minimum distance
    min_distance = np.min(distances)
    return min_distance

def _find_least_overlapping_corner(bbox, bboxes, drawn_boxes, text_size, image_size):
    """Find the corner with the least overlap with other bboxes.
    Args:
        bbox: (y, x, h, w) The bounding box to place the text on.
        bboxes: [(y, x, h, w)] The list of bounding boxes to compare against.
        drawn_boxes: [(y, x, h, w)] The list of bounding boxes that have already been drawn on.
        text_size: (height, width) The size of the text to be drawn.
        image_size: (height, width) The size of the image.
    """
    y, x, h, w = bbox
    h_text, w_text = text_size
    image_height, image_width = image_size
    corners = [
        # top-left
        (y - h_text, x),
        # top-right
        (y - h_text, x + w - w_text),
        # right-top
        (y, x + w),
        # right-bottom
        (y + h - h_text, x + w),
        # bottom-right
        (y + h, x + w - w_text),
        # bottom-left
        (y + h, x),
        # left-bottom
        (y + h - h_text, x - w_text),
        # left-top
        (y, x - w_text),
        ]
    best_corner = corners[0]
    max_flag = float('inf')

    for corner in corners:
        corner_bbox = (corner[0], corner[1], h_text, w_text)
        # if the corner is out of the image, skip
        if corner[0] < 0 or corner[1] < 0 or corner[0] + h_text > image_height or corner[1] + w_text > image_width:
            continue
        max_iou = - (image_width + image_height)
        # 找到关于这个角最差的 case
        # given the current corner, find the larget iou with other bboxes.
        for other_bbox in bboxes + drawn_boxes:
            if np.array_equal(bbox, other_bbox):
                continue
            iou = __calculate_iou(corner_bbox, other_bbox, return_area=True)[1]
            max_iou = max(max_iou, iou - 0.0001 * __calculate_nearest_corner_distance(corner_bbox, other_bbox))
        # the smaller the max_IOU, the better the corner
        # 取最差的值 相对最好的那个角
        if max_iou < max_flag:
            max_flag = max_iou
            best_corner = corner

    return best_corner

def plot_boxes_with_marks(
    image: Image.Image,
    bboxes, # (y, x, h, w)
    mark_helper: MarkHelper,
    linewidth=2,
    alpha=0,
    edgecolor=None,
    fn_save=None,
    normalized_to_pixel=True,
    add_mark=True
) -> np.ndarray:
    """Plots bounding boxes on an image with marks attached to the edges of the boxes where no overlap with other boxes occurs.
    Args:
        image: The image to plot the bounding boxes on.
        bboxes: A 2D int array of shape (num_boxes, 4), where each row represents a bounding box: (y_top_left, x_top_left, box_height, box_width). If normalized_to_pixel is True, the values are float and are normalized with the image size. If normalized_to_pixel is False, the values are int and are in pixel.
    """
    # Then modify the drawing code
    draw = ImageDraw.Draw(image)

    # draw boxes on the image
    image_width, image_height = image.size

    if normalized_to_pixel:
        bboxes = [(int(y * image_height), int(x * image_width), int(h * image_height), int(w * image_width)) for y, x, h, w in bboxes]

    for box in bboxes:
        y, x, h, w = box
        draw.rectangle([x, y, x + w, y + h], outline=edgecolor, width=linewidth)
    
    # Draw the bounding boxes with index at the least overlapping corner
    drawn_boxes = []
    for idx, bbox in enumerate(bboxes):
        text = str(idx)
        text_h, text_w = mark_helper.get_mark_size(text, image_height, image_width)
        corner_y, corner_x = _find_least_overlapping_corner(
            bbox, bboxes, drawn_boxes, (text_h, text_w), (image_height, image_width))
        
        # Define the index box (y, x, y + h, x + w)
        text_box = (corner_y, corner_x, text_h, text_w)

        if add_mark:
            # Draw the filled index box and text
            draw.rectangle([corner_x, corner_y, corner_x + text_w, corner_y + text_h], # (x, y, x + w, y + h)
                        fill="red")        
            font = mark_helper.get_font(image_height, image_width)
            draw.text((corner_x, corner_y), text, fill='white', font=font)
        
        # Update the list of drawn boxes
        drawn_boxes.append(np.array(text_box))
        
    if fn_save is not None: # PIL image
        image.save(fn_save)
    return image

def plot_circles_with_marks(
    image: Image.Image,
    points, # (x, y)
    mark_helper: MarkHelper,
    linewidth=2,
    edgecolor=None,
    fn_save=None,
    normalized_to_pixel=True,
    add_mark=True
) -> np.ndarray:
    """Plots bounding boxes on an image with marks attached to the edges of the boxes where no overlap with other boxes occurs.
    Args:
        image: The image to plot the bounding boxes on.
        bboxes: A 2D int array of shape (num_boxes, 4), where each row represents a bounding box: (y_top_left, x_top_left, box_height, box_width). If normalized_to_pixel is True, the values are float and are normalized with the image size. If normalized_to_pixel is False, the values are int and are in pixel.
    """
    # draw boxes on the image
    image_width, image_height = image.size

    if normalized_to_pixel:
        bboxes = [(int(y * image_height), int(x * image_width), int(h * image_height), int(w * image_width)) for y, x, h, w in bboxes]

    draw = ImageDraw.Draw(image)
    for point in points:
        x, y = point
        draw.circle((x, y), radius=5, outline=edgecolor, width=linewidth)
        
    if fn_save is not None: # PIL image
        image.save(fn_save)
    return image

markhelper = MarkHelper()

BBOX_DEDUPLICATION_IOU_PROPORTION = 0.5
BBOX_GROUPING_VERTICAL_THRESHOLD = 20
BBOX_GROUPING_HORIZONTAL_THRESHOLD = 20
BBOX_AUG_TARGET = 2.0

def _is_boxes_same_line_or_near(bbox1, bbox2, vertical_threshold, horizontal_threshold):
    """check if two boxes are in the same line or close enough to be considered together"""
    y1, x1, h1, w1 = bbox1
    y2, x2, h2, w2 = bbox2
    
    # Check if the boxes are close horizontally (consider the edge case where the boxes are touching)
    horizontally_close = (x1 <= x2 and x2 - x1 <= w1 + horizontal_threshold) or (x2 <= x1 and x1 - x2 <= w2 + horizontal_threshold)

    # Check if the boxes are close vertically (consider the edge case where the boxes are touching)
    vertically_close = (y1 <= y2 and y2 - y1 <= h1 + vertical_threshold) or (y2 <= y1 and y1 - y2 <= h2 + vertical_threshold)
    
    # Consider the boxes to be in the same line if they are vertically close and either overlap or are close horizontally
    return vertically_close and horizontally_close

def _build_adjacency_matrix(bboxes, vertical_threshold, horizontal_threshold):
    """Build the adjacency matrix based on the merging criteria."""
    num_boxes = len(bboxes)
    A = np.zeros((num_boxes, num_boxes), dtype=int)

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            if _is_boxes_same_line_or_near(bboxes[i], bboxes[j], vertical_threshold, horizontal_threshold):
                A[i, j] = 1
                A[j, i] = 1  # Symmetric matrix

    return A

def merge_connected_bboxes(bboxes, text_details, 
    vertical_threshold=BBOX_GROUPING_VERTICAL_THRESHOLD, 
    horizontal_threshold=BBOX_GROUPING_HORIZONTAL_THRESHOLD
):
    """Merge bboxes based on the adjacency matrix and return merged bboxes.
    Args:
        bboxes: A 2D array of shape (num_boxes, 4), where each row represents a bounding box: (y, x, height, width).
        text_details: A list of text details for each bounding box.
        vertical_threshold: The maximum vertical distance between two boxes to be considered in the same line.
        horizontal_threshold: The maximum horizontal distance between two boxes to be considered close.
    """
    # return if there are no bboxes
    if len(bboxes) <= 1:
        return bboxes, text_details
    
    # Convert bboxes (x1, y1, x2, y2) to (y, x, height, width) format
    bboxes = np.array(bboxes)
    bboxes = np.array([bboxes[:, 1], bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1], bboxes[:, 2] - bboxes[:, 0]]).T

    # Build adjacency matrix
    A = _build_adjacency_matrix(bboxes, vertical_threshold, horizontal_threshold)
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(A)
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    # Convert bboxes to (y_min, x_min, y_max, x_max) format
    corners = np.copy(bboxes)
    corners_y, corners_x, corners_h, corners_w = corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]
    
    corners_y_max = corners_y + corners_h
    corners_x_max = corners_x + corners_w
    
    # Merge bboxes for each connected component
    merged_bboxes = []
    merged_text_details = []
    for component in components:
        indices = list(component) # e.g., [32, 33, 34, 30, 31]
        indices = sorted(indices)

        # merge the text details
        merged_text_details.append(' '.join([text_details[i] for i in indices]))

        # merge the bboxes
        y_min = min(corners_y[i] for i in indices)
        x_min = min(corners_x[i] for i in indices)
        y_max = max(corners_y_max[i] for i in indices)
        x_max = max(corners_x_max[i] for i in indices)
        merged_bboxes.append((y_min, x_min, y_max - y_min, x_max - x_min)) # Convert merged_bbox back to (y, x, height, width) format
    
    # convert (y, x, height, width) to (x1, y1, x2, y2) format without np.array
    merged_bboxes = [(bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]) for bbox in merged_bboxes]
    return merged_bboxes, merged_text_details