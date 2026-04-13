import json
import os

import torch
import yaml
from datasets import load_dataset

from .openx import OpenXDataItem


class DataItem:
    """
    Curate data items from all data sources
    """
    def __init__(self, training_size=-1, local_run=False):
        self.training_size = training_size
        self.local_run = local_run

    def _get_dataset_tag(self, data_path):
        if "epic" in data_path.lower():
            return "epic"
        elif "open-x" in data_path or "openx" in data_path:
            if 'traces' in data_path:
                return "openx_magma"
            else:
                return "openx"
        elif "sthv2" in data_path.lower():
            return "sthv2"
        elif "exoego4d" in data_path.lower():
            return "exoego4d"
        elif 'ego4d' in data_path.lower():
            return "ego4d"
        elif 'aitw' in data_path.lower():
            return "aitw"
        elif 'seeclick' in data_path.lower() and 'ocr' in data_path.lower():
            return "seeclick_ocr"            
        elif 'seeclick' in data_path.lower():
            return "seeclick"
        elif 'mind2web' in data_path.lower():
            return "mind2web"
        elif 'vision2ui' in data_path.lower():
            return "vision2ui"
        elif 'llava' in data_path.lower():
            return "llava"
        elif 'magma' in data_path.lower():
            return "magma"
        elif 'sharegpt4v' in data_path.lower():
            return "sharegpt4v"
        else:
            raise ValueError(f"Dataset tag not found for {data_path}")
    
    def _get_items(self, data_path, image_folder=None, processor=None, conversation_lib=None):
        if data_path.endswith(".json"):
            list_data_dict = json.load(open(data_path, "r"))
        elif data_path.endswith(".jsonl"):
            list_data_dict = [json.loads(line) for line in open(data_path, "r")]
        elif data_path.endswith(".pth"):
            list_data_dict = torch.load(data_path, map_location="cpu")
            # random.shuffle(list_data_dict)
        else:
            if self._get_dataset_tag(data_path) == "openx":
                list_data_dict = OpenXDataItem()(data_path, image_folder, processor=processor, conversation_lib=conversation_lib, local_run=self.local_run)
            elif self._get_dataset_tag(data_path) == "pixelprose":
                # Load the dataset
                list_data_dict = load_dataset(
                    data_path, 
                    cache_dir=image_folder
                )
            else:
                data_folder = os.path.dirname(data_path)
                # get file name from data_path
                data_files = data_path.split('/')[-1].split('+')
                list_data_dict = []
                for file in data_files:
                    json_path = os.path.join(data_folder, file + '.json')      
                    list_data_dict.extend(json.load(open(json_path, "r")))                
        return list_data_dict
    
    def __call__(self, data_path, processor=None, conversation_lib=None, is_eval=False):
        assert data_path is not None, "Data path is not provided"
        if data_path.endswith(".yaml"):
            data_dict = yaml.load(open(data_path, "r"), Loader=yaml.FullLoader)    
            data_path_key = 'DATA_PATH' if not is_eval else 'DATA_PATH_VAL'
            image_folder_key = 'IMAGE_FOLDER' if not is_eval else 'IMAGE_FOLDER_VAL'
            assert len(data_dict[data_path_key]) == len(data_dict[image_folder_key]), "Data path and image folder mismatch"
            items = {}
            dataset_names = []
            dataset_folders = []
            for i, (data_path, image_folder) in enumerate(zip(data_dict[data_path_key], data_dict[image_folder_key])):
                items_temp = self._get_items(data_path, image_folder, processor, conversation_lib)                
                dataset_tag = self._get_dataset_tag(data_path)                
                if dataset_tag != "openx":
                    # if self.training_size > 0:
                    #     items_temp = items_temp[:self.training_size]             
                    if dataset_tag in ['sthv2', "ego4d", "exoego4d"]: 
                        for item in items_temp:
                            item['image_folder'] = image_folder
                            item['dataset_tag'] = dataset_tag
                            item['gpt_response'] = ''
                            item['global_instructions'] = item['annotations']
                    elif dataset_tag in ["openx_magma"]:
                        items_dict_temp = []
                        for item in items_temp:
                            items_dict_temp.append(
                                {
                                    'image': item.replace('traces', 'images').replace('.pth', '.jpg'),
                                    'trace': item,
                                    'image_folder': image_folder,
                                    'dataset_tag': dataset_tag
                                }
                            ) 
                        items_temp = items_dict_temp         
                    else:
                        # add image_foler to each item
                        for item in items_temp:
                            item['image_folder'] = image_folder
                        # add dataset tag to each item
                        for item in items_temp:
                            item['dataset_tag'] = dataset_tag
                if dataset_tag in items:
                    items[dataset_tag].extend(items_temp)
                else:
                    items[dataset_tag] = items_temp
                    dataset_names.append(dataset_tag)
                    dataset_folders.append(image_folder)
        else:
            items = self._get_items(data_path)
            dataset_names = None
            dataset_folders = None  
        return items, dataset_names, dataset_folders