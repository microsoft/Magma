import collections
import copy
import os
import random
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from data.utils.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
)
from magma.processing_magma import MagmaProcessor

from . import *
from .data_item import DataItem


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: None
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # move all DEFAULT_IMAGE_TOKEN to the beginning of the sentence
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                # count the number of DEFAULT_IMAGE_TOKEN in the sentence
                num_image_tokens = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                # remove all DEFAULT_IMAGE_TOKEN from the sentence
                if data_args.mm_use_image_start_end and (DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN) in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN +'\n', '').replace(DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN, '')
                else:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN + '\n', '').replace(DEFAULT_IMAGE_TOKEN, '')
                # add num_image_tokens DEFAULT_IMAGE_TOKEN to the beginning of the sentence
                sentence['value'] = (DEFAULT_IMAGE_TOKEN + '\n') * num_image_tokens + sentence['value']
                if data_args.mm_use_image_start_end:
                    sentence['value'] = sentence['value'].replace('<image>', DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN)                     

    return sources

def preprocess(
    sources: Sequence[str],
    processor: MagmaProcessor, 
    has_image: bool = False):

    conversations = []
    for i, source in enumerate(sources):
        convs = copy.deepcopy(source)
        for elem in convs:
            elem['role'] = 'user' if elem['from'] in ['human', 'user'] else 'assistant'
            elem['content'] = elem['value']
        convs = [
            {
                "role": "system",
                "content": "You are agent that can see, talk and act.", 
            },            
        ] + convs
        
        text = processor.tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=False
        )
        conversations.append(text)

    # NOTE: this is only for QWen
    # get the sep1 and sep2
    dummy_convs = [        
        {
            "role": "system",
            "content": "You are agent that can see, talk and act.", 
        },
        {
            "role": "user",
            "content": ""
        },
        {
            "role": "assistant",
            "content": ""
        }
    ]
    dummy_text = processor.tokenizer.apply_chat_template(
        dummy_convs,
        tokenize=False,
        add_generation_prompt=False,
    )    

    empty_token_lengh = len(processor.tokenizer("").input_ids)

    bos_token = processor.tokenizer.bos_token
    eos_token = processor.tokenizer.eos_token
    if 'phi' in processor.tokenizer.name_or_path.lower():
        eos_token = '<|end|>\n'
    elif 'qwen2-' in processor.tokenizer.name_or_path.lower():
        bos_token = '<|im_start|>'
        eos_token = '<|im_end|>\n'

    segments = dummy_text.split(eos_token)[:-1]
    sep1, sep2 = segments[-2], segments[-1]
    if bos_token:
        sep1 = sep1.replace(bos_token, '')

    tokenizer = processor.tokenizer
    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids

    targets = input_ids.clone()

    for k, (conversation, target) in enumerate(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        if 'phi' in processor.tokenizer.name_or_path.lower():
            # Phi-3 has an pad_token at the end
            total_len = total_len + 1
        
        conversation_sys = conversation.split(sep1)[0]
        conversation = conversation[len(conversation_sys):]
        rounds = conversation.split(sep1)[1:]
        cur_len = len(tokenizer(conversation_sys).input_ids)
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep2)
            if len(parts) != 2:
                break
            parts[0] = sep1 + parts[0] + sep2
            rou = sep1 + rou
            
            # NOTE: the reason to minus 1 is because tokenizer will give a start token, e.g., 128000 for llama3
            round_len = len(tokenizer(rou).input_ids) - empty_token_lengh
            instruction_len = len(tokenizer(parts[0]).input_ids) - empty_token_lengh

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
                print(
                    conversations[k]
                )
    return dict(
        input_ids=input_ids,
        labels=targets,
    )

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                processor: MagmaProcessor, 
                data_items: Dict,
                dataset_names: List[str],
                dataset_folders: List[str],
                data_args: None):
        super(LazySupervisedDataset, self).__init__()
        self.processor = processor
        self.data_args = data_args
        self.data_items = []
        self.conv_constructor = {}
        if dataset_names is not None:
            for dataset_name, dataset_folder in zip(dataset_names, dataset_folders):
                if dataset_name in ['sharegpt4v', 'aitw', 'mind2web']:
                    self.data_items.extend(data_items[dataset_name])
                elif dataset_name in ['seeclick', 'vision2ui', 'seeclick_ocr']:
                    self.conv_constructor[dataset_name] = eval(dataset_name)(
                        mm_use_trace_start_end=data_args.mm_use_trace_start_end,
                        mm_use_trace_speed=data_args.mm_use_trace_speed,
                        mm_use_image_start_end=data_args.mm_use_image_start_end,
                        mm_use_image_history=data_args.mm_use_image_history,
                        mm_use_som_tom=data_args.mm_use_som_tom,
                        mm_use_som_tom_orig_img=data_args.mm_use_som_tom_orig_img,
                        remove_static_trace_pts=data_args.remove_static_trace_pts,
                        spatial_quant_size=data_args.spatial_quant_size,                    
                        dataset_folder=dataset_folder, 
                        show_trace=data_args.show_trace,
                        task=data_args.task,
                        training_size=data_args.training_size,
                        tokenizer=processor.tokenizer,
                    )
                    final_items = self.conv_constructor[dataset_name].filter_items(data_items[dataset_name])
                    self.data_items.extend(final_items)                    
                else:
                    self.conv_constructor[dataset_name] = eval(dataset_name)(
                        mm_use_trace_start_end=data_args.mm_use_trace_start_end,
                        mm_use_trace_speed=data_args.mm_use_trace_speed,
                        mm_use_image_start_end=data_args.mm_use_image_start_end,
                        mm_use_image_history=data_args.mm_use_image_history,
                        mm_use_som_tom=data_args.mm_use_som_tom,
                        mm_use_som_tom_orig_img=data_args.mm_use_som_tom_orig_img,
                        remove_static_trace_pts=data_args.remove_static_trace_pts,
                        spatial_quant_size=data_args.spatial_quant_size,                    
                        dataset_folder=dataset_folder, 
                        show_trace=data_args.show_trace,    
                        task=data_args.task,
                        training_size=data_args.training_size,
                        tokenizer=processor.tokenizer,
                    )
                    final_items = self.conv_constructor[dataset_name].filter_items(data_items[dataset_name])
                    self.data_items.extend(final_items)
        self.action_placeholder_token_id = self.processor.tokenizer.convert_tokens_to_ids('<action>')

    def __len__(self):
        return len(self.data_items)

    @property
    def lengths(self):
        length_list = []
        for sample in self.data_items:
            img_tokens = 128 if ('image' in sample and sample['image'] is not None) else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data_items:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample and sample['image'] is not None) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = copy.deepcopy(self.data_items[i])
        if 'video' in item and item['video'][0] is not None:
            assert item['image_folder'] is not None or self.data_args.image_folder is not None, "image_folder is not provided"
            image_folder = self.data_args.image_folder if self.data_args.image_folder is not None else item['image_folder']
            if item['dataset_tag'] in ['sthv2', 'ego4d', 'exoego4d']:
                visual_trace_path = os.path.join(image_folder, item['trace'])
                if os.path.exists(visual_trace_path):
                    try:
                        visual_traces = torch.load(visual_trace_path, map_location='cpu')
                        video_path = os.path.join(image_folder, item['video'].replace('/home/tanreuben/vlp_datasets/', ''))
                        item.update(visual_traces)
                    except Exception as e:
                        print(f"Error loading: {visual_trace_path}")
                        visual_traces = None
                        video_path = None
                else:                
                    print(f"Error: {visual_trace_path} not found")    
                    visual_traces = None       
                    video_path = None
                item = self.conv_constructor[item['dataset_tag']](item=item, video_path=video_path, visual_traces=visual_traces)                                              
            else:
                item['video'][0] = item['video'][0].replace('/mnt/data/video_datasets_visual_traces/YouCook2/', '')
                video_path = os.path.join(image_folder, item['video'][0])
                frame_start, frame_end = item['frame_interval'][0].item(), item['frame_interval'][1].item()
                video_name = os.path.basename(video_path).split('.')[0]
                if 'youcook2' in video_path.lower():
                    visual_trace_path = os.path.join(image_folder, 'all_detected_visual_traces_30fps', f'{video_name}_trace_{frame_start:09d}_{frame_end:09d}.pth')
                else:
                    visual_trace_path = os.path.join(image_folder, 'visual_trace' if 'epic' in image_folder else 'visual_traces', video_name, f'trace_{frame_start:09d}_{frame_end:09d}.pth')
                if os.path.exists(visual_trace_path):
                    visual_traces = torch.load(visual_trace_path, map_location='cpu')
                else:
                    visual_traces = None
                item = self.conv_constructor[item['dataset_tag']](item=item, video_path=video_path, visual_traces=visual_traces)    
            image = item['image']
            num_crops = item['num_crops']
            # if image is not a PIL image
            if image is None:  
                base_img_size = self.processor.image_processor.base_img_size
                image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))
                item['image'] = image
                num_crops = 1
            image_pt = self.processor.image_processor(image, num_crops=num_crops, return_tensors='pt')
            images = collections.defaultdict(list)
            for key, val in image_pt.items():
                images[key].append(val)
            texts = [item["conversations"]]
        elif 'image' in item and item['image'] is not None:
            import pdb; pdb.set_trace()
            # cope with multiple images
            image_folder = item['image_folder']
            image_files = item['image']
            if isinstance(image_files, str):
                image_files = [image_files]
            image_files = [image_path.replace("ming2web_images", "mind2web_images") for image_path in image_files]
            # image_files = image_files*2       
            # item["conversations"][0]['value'] = item["conversations"][0]['value'] + '\n' + DEFAULT_IMAGE_TOKEN
            images = collections.defaultdict(list)
            for image_file in image_files:
                image_file = image_file[1:] if image_file.startswith('/') else image_file
                image_path = os.path.join(image_folder, image_file)
                try:
                    if "trace" in self.data_items[i]:
                        trace_file = self.data_items[i]["trace"]
                        trace_path = os.path.join(image_folder, trace_file)
                        if os.path.exists(trace_path):
                            visual_traces = torch.load(trace_path, map_location='cpu')
                            item.update(visual_traces)
                        else:
                            visual_traces = None
                        video_path = image_path
                        item = self.conv_constructor[item['dataset_tag']](item=item, video_path=image_path, visual_traces=visual_traces)               
                        image = item['image_data']
                        num_crops = item['num_crops']
                        if image is None:  
                            base_img_size = self.processor.image_processor.base_img_size
                            image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))
                            num_crops = 1
                        # NOTE: override num_crops for robotics dataset
                        image = self.processor.image_processor(image, num_crops=num_crops, return_tensors='pt')
                    elif 'ocrs' in self.data_items[i]:
                        item = self.conv_constructor[item['dataset_tag']](item=item)        
                        image = self.processor.image_processor(image, return_tensors='pt')
                    else:
                        # regular image sft dataset
                        image = Image.open(image_path).convert('RGB')           
                        # if item['dataset_tag'] in ['seeclick', 'vision2ui']:
                        #     image = self.processor.image_processor(image, num_crops=9, return_tensors='pt')
                        # else:                   
                        image = self.processor.image_processor(image, return_tensors='pt')
                    for key, val in image.items():
                        images[key].append(val)
                except Exception as e:
                    print(f"Error: {e}")
                    base_img_size = self.processor.image_processor.base_img_size
                    image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))                    
                    image = self.processor.image_processor(image, num_crops=1, return_tensors='pt')
                    for key, val in image.items():
                        images[key].append(val)
            texts = preprocess_multimodal(
                copy.deepcopy([item["conversations"]]),
                self.data_args)
        else:
            images = collections.defaultdict(list)
            # image does not exist in the data, but the model is multimodal
            base_img_size = self.processor.image_processor.base_img_size
            image = Image.new('RGB', (base_img_size, base_img_size), (0, 0, 0))
            image = self.processor.image_processor(image, num_crops=1, return_tensors='pt')
            for key, val in image.items():
                images[key].append(val)
            item["conversations"][0]['value'] = DEFAULT_IMAGE_TOKEN + '\n' + item["conversations"][0]['value']
            if self.data_args.mm_use_image_start_end:
                item["conversations"][0]['value'] = item["conversations"][0]['value'].replace('<image>', DEFAULT_IM_START_TOKEN + '<image>' + DEFAULT_IM_END_TOKEN)
            texts = [item["conversations"]]
        
        data_dict = preprocess(
            texts,
            self.processor,
            has_image=('image' in item and item['image'] is not None)
        )

        if self.action_placeholder_token_id in data_dict['input_ids']:
            assert (data_dict['input_ids'] == self.action_placeholder_token_id).sum() == 7, "action token length should be 7 in input_ids"
            assert self.action_placeholder_token_id in data_dict['labels'], "action token should be also in labels"
            assert (data_dict['labels'] == self.action_placeholder_token_id).sum() == 7, "action token length should be 7 in labels"
            # replace the action token with the actual action token item['action_token_ids']
            action_token_ids = torch.tensor(item['action_token_ids'], dtype=torch.long)[None,:]
            data_dict['input_ids'][data_dict['input_ids'] == self.action_placeholder_token_id] = action_token_ids     
            data_dict['labels'][data_dict['labels'] == self.action_placeholder_token_id] = action_token_ids                   
        
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict.update(images)
        data_dict.update(
            {
                "dataset_name": item["dataset_tag"],
                "item_id": i
            }
        )
        del item
        return data_dict


# Custom wrapper to combine Dataset and IterableDataset without loading IterableDataset in memory
class CombinedDataset(Dataset):
    def __init__(self, dataset, iterable_dataset, local_run=False, seed=7):
        self.dataset_len = []
        if dataset is not None:
            self.dataset_len.append(len(dataset)) # Length of the Dataset   
            if dist.is_initialized():
                sampler = DistributedSampler(
                    dataset, 
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    seed=seed,
                    drop_last=False,
                )
            else:
                sampler = None            
            self.iterable_dataset_a = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0 if local_run else 8, pin_memory=False)  # DataLoader for the Dataset
            self.iterable_iter_a = iter(self.iterable_dataset_a)
        else:
            self.iterable_dataset_a = None
            self.iterable_iter_a = None
            self.dataset_len.append(0)

        if iterable_dataset is not None:  
            self.dataset_len.append(len(iterable_dataset)) # Length of the IterableDataset   
            self.iterable_dataset_b = iterable_dataset
            self.iterable_iter_b = iter(self.iterable_dataset_b)  # Iterator for the IterableDataset
        else:
            self.iterable_dataset_b = None
            self.iterable_iter_b = None
            self.dataset_len.append(0)        
        self.sampling_ratios = [float(item)/sum(self.dataset_len) for item in self.dataset_len]
        print(f"total training data size: {sum(self.dataset_len)}")
        print(f"sampling ratios: {self.sampling_ratios}")

    def __len__(self):
        # Length can be the maximum of both or some other logic
        return sum(self.dataset_len)
    
    def __getitem__(self, index):
        # according to the sampling ratio, choose which dataset to sample
        dataset_choice = random.choices([0, 1], self.sampling_ratios)[0]
        if dataset_choice == 0:
            # Fetch a sample from the IterableDataset using its iterator
            try:
                iterable_sample_a = next(self.iterable_iter_a)
            except StopIteration:
                # Reinitialize the iterator if it exhausts
                self.iterable_iter_a = iter(self.iterable_dataset_a)
                iterable_sample_a = next(self.iterable_iter_a)
            iterable_sample_a['input_ids'] = iterable_sample_a['input_ids'][0]
            iterable_sample_a['labels'] = iterable_sample_a['labels'][0]
            iterable_sample_a['pixel_values'] = [item[0] for item in  iterable_sample_a['pixel_values']]
            iterable_sample_a['image_sizes'] = [item[0] for item in  iterable_sample_a['image_sizes']]
            return iterable_sample_a
        else:
            # Fetch a sample from the IterableDataset using its iterator
            try:
                iterable_sample_b = next(self.iterable_iter_b)
            except StopIteration:
                # Reinitialize the iterator if it exhausts
                self.iterable_iter_b = iter(self.iterable_dataset_b)
                iterable_sample_b = next(self.iterable_iter_b)
            # print(f"oxe-rank-{rank}: {iterable_sample_b['dataset_name']}")
            # Return a combined sample (modify based on your requirement)
            return iterable_sample_b

def build_joint_dataset(
        data_path: str,
        processor: MagmaProcessor,
        data_args: None, 
        is_eval: bool = False
    ) -> torch.utils.data.ConcatDataset:

    data_items, dataset_names, dataset_folders = DataItem(training_size=data_args.training_size, local_run=data_args.local_run)(data_path, processor, None, is_eval=is_eval)
    # pop out open-x dataset
    openx_dataset = None
    if 'openx' in data_items:
        openx_dataset = data_items.pop('openx')
        _ = dataset_folders.pop(dataset_names.index('openx'))
        _ = dataset_names.pop(dataset_names.index('openx'))

        lazy_dataset = None
        if len(data_items) > 0:
            lazy_dataset = LazySupervisedDataset(processor, data_items, dataset_names, dataset_folders, data_args)

        # concatenate openx dataset and lazy_dataset
        return CombinedDataset(lazy_dataset, openx_dataset, local_run=data_args.local_run)
    else:
        return LazySupervisedDataset(processor, data_items, dataset_names, dataset_folders, data_args)

