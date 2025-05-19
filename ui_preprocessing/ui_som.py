import torch
import torchvision
import pandas as pd
import pyarrow
import json
from bs4 import BeautifulSoup
from agents.ui_agent.util.som import MarkHelper
from agents.ui_agent.util.utils import get_som_labeled_img
from PIL import Image
import os # needed for file/path handling

input_path = "/mnt/hitchcock/scratch/oberon/Multimodal-Mind2Web/exp1_data"
output_path = "/mnt/hitchcock/scratch/oberon/Multimodal-Mind2Web/prepos_data"
output_image_dir = os.path.join(output_path, "marked_images") # specifc folder for images
output_annotations_dir = os.path.join(output_path, "annotations") # specific folder for json

os.makedirs(output_image_dir, exist_ok = True)
os.makedirs(output_annotations_dir, exist_ok = True)

output_json_filepath = os.path.join(output_annotations_dir, "annotations.json")

mark_helper = MarkHelper()

all_processed_records = []

for filename in os.listdir(input_path):
    try:
        if filename.endswith(".parquet") and filename.startswith("train"):
            parquet_file_path = os.path.join(input_path, filename)
            print(f"Processing file: {parquet_file_path}")
            try:
                data_frame = pd.read_parquet(parquet_file_path)

                for index, row in data_frame.iterrows():
                    print(f"Processing sample ID: {row.get('action_uid', index)}")

                    try:
                        image_relative_path = row['screenshot']

                        image_full_path = os.path.join(input_path, "...", "screenshots", image_relative_path)

                        image = Image.open(image_full_path).convert("RGB")
                    
                    except Exception as e:
                        #print(f"Error loading image {image_relative_path}: {e}")
                        continue
                    
                    pos_candidates = row['pos_candidates']
                    target_element_info = None
                    for candidate in pos_candidates:
                        if candidate.get('is_original_target', False) or candidate.get('is_top_level_target', False):
                            target_element_info = candidate
                            break

                    if not target_element_info:
                        print(f"Warning: No target element found for sample {row.get('action_uid', index)}")
                        continue

                    # TODO
                    print("Placeholder: implement HTML parsing to find target coordinates")
                    target_bbox = [10, 10, 50, 50] # dummy coordinates for test

                    # TODO
                    print("Placeholder: implement detection/parsing for all elements")
                    all_bboxes_for_marking = [target_bbox, [100, 100, 150, 120]]

                    def xyxy_to_yxhw(box):
                        y = box[1]
                        x = box[0]
                        h = box[3] - box[1]
                        w = box[2] - box[0]
                        return [y, x, h, w]
                    
                    bboxes_yxhw = [xyxy_to_yxhw(box) for box in all_bboxes_for_marking]

                    target_index_in_list = -1

                    for i, box in enumerate(all_bboxes_for_marking):
                        if box == target_bbox:
                            target_index_in_list = i
                            break

                    if target_index_in_list == -1:
                        print(f"Error: target bounding box not found in list for marking {row.get('action_uid', index)}")
                        continue

                    SoM_Target_Output = target_index_in_list

                    marked_image = plot_boxes_with_marks(
                        image.copy(),
                        bboxes_yxhw,
                        mark_helper = mark_helper,
                        edgecolor = (255, 0, 0)
                    )

                    marked_image_filename = f"{row['action_uid']}.png"
                    marked_image_path = os.path.join(output_image_dir, marked_image_filename)
                    marked_image.save(marked_image_path)

                    instruction = row['confirmed_task']
                    output_record = {
                        "id": row['action_uid'],
                        "image": os.path.relpath(marked_image_path, output_path),
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{instruction}"},
                            {"from": "agent", "value": f"Mark: {SoM_Target_Output + 1}"}
                        ]
                    }
                    all_processed_records.append(output_record)

            except Exception as e:
                #print(f"Error processing file {filename}: {e}")
                print()
            
    except Exception as e:
        #print(f"Error opening file {filename}: {e}. File is {type(filename)}")
        print()

print(f"Saving {len(all_processed_records)} processed records to {output_json_filepath}")
with open(output_json_filepath, 'w') as f:
    json.dump(all_processed_records, f, indent = 2)

print("Preprocessing script finished.")
