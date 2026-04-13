from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from data.utils.constants import (
    IGNORE_INDEX,
)
from magma.processing_magma import MagmaProcessor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    processor: MagmaProcessor

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:       
        input_ids, labels, pixel_values, image_sizes = \
            tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "pixel_values", "image_sizes"))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.processor.tokenizer.model_max_length]
        labels = labels[:, :self.processor.tokenizer.model_max_length]

        pixel_values = [torch.cat(pv, dim=0) for pv in pixel_values]
        image_sizes = [torch.cat(isz, dim=0) for isz in image_sizes]
        pixel_values_padded = torch.nn.utils.rnn.pad_sequence(pixel_values, batch_first=True, padding_value=0)
        image_sizes_padded = torch.nn.utils.rnn.pad_sequence(image_sizes, batch_first=True, padding_value=0)
        
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.processor.tokenizer.pad_token_id),
            pixel_values=pixel_values_padded,
            image_sizes=image_sizes_padded
        )
        return batch

@dataclass
class DataCollatorForHFDataset(object):
    """Collate hugging face examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0] and instances[0]['image'] is not None:
            images = [instance['image'] for instance in instances]
            # if all(x is not None and x.shape == images[0].shape for x in images):
            #     batch['images'] = torch.stack(images)
            # else:
            batch['images'] = images

        if 'add_im_loss' in instances[0]:
            batch['add_im_loss'] = True
        if 'max_num_crops' in instances[0]:
            batch['max_num_crops'] = instances[0]['max_num_crops']
        return batch