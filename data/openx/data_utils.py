
import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import yaml

from data.conversations import Constructor
from data.utils.visual_tracker import visual_tracker

from .conf import VLAConfig
from .materialize import get_vla_dataset_and_collator

logger = logging.getLogger(__name__)

"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

@dataclass
class OpenXDataItem:
    def __call__(self, data_root_dir, data_soup, processor=None, conversation_lib=None, image_aug=False, local_run=False, future_action_window_size=1):
        # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
        self.openx_data_cfg = VLAConfig.get_choice_class(data_soup)
        default_image_resolution = processor.image_processor.base_img_size
        logger.info(f"Creating VLA Open-X Dataset with Mixture `{self.openx_data_cfg.data_mix}`")
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)       

        # get local rank for distributed training
        rank = dist.get_rank() if dist.is_initialized() else 0        
        rank = rank % torch.cuda.device_count()
        openx_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
            data_root_dir,
            self.openx_data_cfg.data_mix,
            shuffle_buffer_size=1 if (local_run or future_action_window_size>1) else self.openx_data_cfg.shuffle_buffer_size,
            image_transform=processor.image_processor,
            visual_tracker=visual_tracker(**self.settings.get('tracker', None), device=f"cuda:{rank}"), 
            dataset_settings=self.settings,
            tokenizer=processor.tokenizer,
            default_image_resolution=(3, default_image_resolution, default_image_resolution),
            image_aug=image_aug,
            future_action_window_size=future_action_window_size, 
            prompt_builder_fn=conversation_lib, # vlm.llm_backbone.prompt_builder_fn,
            local_run=local_run,
        )

        # Save dataset statistics for de-normalization at inference time
        # if overwatch.is_rank_zero():
        #     save_dataset_statistics(openx_dataset.dataset_statistics, run_dir)

        return openx_dataset

class OpenX(Constructor):
    def __init__(self, **kwargs):
        super(OpenX, self).__init__(**kwargs)
        # load settings from settings.yaml file
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.yaml'), 'r') as file:
            self.settings = yaml.safe_load(file)
        self.spatial_quant_size = kwargs.get('spatial_quant_size', 256)   # this is also used for open-x
        self.num_clusters = self.settings['trace_processor']['num_clusters']
        self.root_dir = kwargs.get('dataset_folder', None)

    def __call__(self, **kwargs):
        return super()._construct_conv(**kwargs)
    
    def filter_items(self, items):
        """
        Filter invalid items
        """
        return items
