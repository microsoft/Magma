import os
import re

import yaml
from tqdm import tqdm

from data.conversations import Constructor


class SeeClick(Constructor):
    def __init__(self, **kwargs):
        super(SeeClick, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.task = kwargs.get('task', 'agent')
        self.use_som_tom = kwargs.get('mm_use_som_tom', True)
        self.use_som_tom_orig_img = kwargs.get('mm_use_som_tom_orig_img', False)

    def __call__(self, **kwargs):
        return super()._construct_conv(**kwargs)
    
    def filter_items(self, items):
        """
        Filter invalid items
        """
        if self.use_som_tom and not self.use_som_tom_orig_img:
            return items        
        elif self.use_som_tom and self.use_som_tom_orig_img:
            print("Adding original image to SoM")
            for item in tqdm(items):
                image_path = item['image']
                if "mobile"  in image_path:
                    item['image'] = [image_path.replace("combined_image_processed", "combined")] + [item['image']]
                    for conv in item['conversations']:
                        # remove 'Mark: {id}' from the conversation conv['value'], e.g., Mark: 11
                        conv['value'] = conv['value'].replace("<image>", "<image>\n<image>")
                elif "web" in image_path:
                    item['image'] = [image_path.replace("seeclick_web_imgs_processed", "seeclick_web_imgs")] + [item['image']]
                    for conv in item['conversations']:
                        # remove 'Mark: {id}' from the conversation conv['value']
                        conv['value'] = conv['value'].replace("<image>", "<image>\n<image>")
                else:
                    continue   
        else:
            print("Filtering SoM from seeclick training data")
            for item in tqdm(items):
                image_path = item['image']
                if "mobile"  in image_path:
                    item['image'] = image_path.replace("combined_image_processed", "combined") 
                    for conv in item['conversations']:
                        # remove 'Mark: {id}' from the conversation conv['value'], e.g., Mark: 11
                        conv['value'] = re.sub(r' Mark: \d+', '', conv['value']).strip()
                        conv['value'] = re.sub(r' mark: \d+', '', conv['value']).strip()
                elif "web" in image_path:
                    item['image'] = image_path.replace("seeclick_web_imgs_processed", "seeclick_web_imgs")
                    for conv in item['conversations']:
                        # remove 'Mark: {id}' from the conversation conv['value']
                        conv['value'] = re.sub(r' Mark: \d+', '', conv['value']).strip()
                        conv['value'] = re.sub(r' mark: \d+', '', conv['value']).strip()
                else:
                    continue         
        return items