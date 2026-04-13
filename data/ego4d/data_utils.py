import logging
import os

import yaml
from tqdm import tqdm

from data.conversations import Constructor

logger = logging.getLogger(__name__)

class Ego4d(Constructor):
    def __init__(self, **kwargs):
        super(Ego4d, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.task = kwargs.get('task', 'agent')
        self.use_som_tom = kwargs.get('mm_use_som_tom', True)

        if kwargs.get('training_size', 'default') == 'default':
            self.training_size = self.settings['training'].get('size', -1)
        else:
            self.training_size = kwargs.get('training_size', -1)
            # convert M to 1000000, e.g, 10M means 10,000,000
            if 'M' in self.training_size:
                self.training_size = int(float(self.training_size.replace('M', '')) * 1000000)
            else:
                self.training_size = int(self.training_size)

        self.filtered_verb = [
            'converse',
            'walk',
            'laugh',
            'stand',
            'move around',
            'looks around', 
        ]
    def __call__(self, **kwargs):
        return super()._construct_conv(**kwargs)
    
    def filter_items(self, items):
        """
        Filter invalid items
        """
        filtered_items = []
        print("Filtering items")
        for item in tqdm(items):
            global_instruction = item['global_instructions']
            if len(global_instruction) == 0:
                continue
            # check if global_instruction contain any word in self.filtered_verb
            # if so, skip this item
            if any(verb in global_instruction for verb in self.filtered_verb):           
                continue
            seg_name = item['video'].split('/')[-1]
            start_str, end_str = seg_name.split('___')[0:2]
            start_time = float(start_str.split('_')[-1])
            end_time = float(end_str.split('_')[-1])
            if (end_time-start_time) < 1:
                continue
            filtered_items.append(item)        
        if self.training_size > 0 and self.training_size < len(filtered_items):
            # sample uniformly self.training_size samples from the filtered items
            filtered_items = filtered_items[::(len(filtered_items)//self.training_size)]
        print(f"Keep {len(filtered_items)} items from {len(items)} items")
        return filtered_items