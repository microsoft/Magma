import os

import yaml

from data.conversations import Constructor


class EpicKitchen(Constructor):
    def __init__(self, **kwargs):
        super(EpicKitchen, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)
        self.task = kwargs.get('task', 'agent')
        self.use_som_tom = kwargs.get('mm_use_som_tom', True)

    def __call__(self, **kwargs):
        if self.task == "captioner":
            return super()._construct_caption(**kwargs)
        else:
            return super()._construct_conv(**kwargs)
    
    def filter_items(self, items):
        """
        filter out items that are not suitable for conversation construction
        """
        filtered_items = []
        for item in items:
            # remove closeup videos
            if 'closeup' in item['gpt_response'][0] or \
                'close-up' in item['gpt_response'][0] or \
                    'close up' in item['gpt_response'][0] or \
                        'What you should do next' not in item['gpt_response'][0]:
                continue
            # item['gpt_response'][0] = item['gpt_response'][0].replace('blue', 'yellow')
            filtered_items.append(item)
        print(f"Filtered {len(items) - len(filtered_items)} items from {len(items)} items")
        return filtered_items