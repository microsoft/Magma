import os

import yaml
from tqdm import tqdm

from data.conversations import Constructor


class Magma(Constructor):
    def __init__(self, **kwargs):
        super(Magma, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.task = kwargs.get('task', 'agent')
        self.use_som_tom = kwargs.get('mm_use_som_tom', True)
        self.tokenizer = kwargs.get('tokenizer', None)
        self.special_tokens = [self.tokenizer.pad_token]

    def __call__(self, **kwargs):
        return super()._construct_conv(**kwargs)
    
    def filter_items(self, items):
        """
        Filter invalid items
        """
        num_items = len(items)
        print("Filtering samples containing special tokens")
        for item in tqdm(items):
            values = [conv['value'] for conv in item['conversations']]
            # if any special token is present in the conversation, remove the item
            if any([True for value in values if any([token in value for token in self.special_tokens])]):
                print(item)
                items.remove(item)
        print(f"Removed {num_items - len(items)} items containing special tokens")
        return items