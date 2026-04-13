# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from transformers.trainer import get_model_param_count

from data import *
from magma.configuration_magma import MagmaConfig
from magma.image_processing_magma import MagmaImageProcessor
from magma.modeling_magma import MagmaForCausalLM
from magma.processing_magma import MagmaProcessor
from trainer import MagmaTrainer

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/Magma-8B")
    version: Optional[str] = field(default="magma_instruct")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_ckpt: Optional[str] = field(default=None)
    img_anyres_strategy: Optional[str] = field(default='crop')
    proj_vis_to_txt_tokens: bool = field(default=False)
    img_size: Optional[int] = field(default=640)   # default to the last layer
    vision_backbone: Optional[str] = field(default="convnextlarge")
    tune_vision_tokenizer: Optional[str] = field(default='none')
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_trace_start_end: bool = field(default=False)
    mm_use_trace_speed: bool = field(default=False)
    mm_use_image_start_end: bool = field(default=False)
    mm_use_image_history: bool = field(default=False)
    mm_use_som_tom: bool = field(default=True)
    mm_use_som_tom_orig_img: bool = field(default=False)
    spatial_quant_size: Optional[int] = field(default=256)
    remove_static_trace_pts: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    flash_attn_2_enabled: bool = False
    task: Optional[str] = field(default="agent")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    data_format: str = "llava"
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    max_num_crops: int = 25
    add_im_loss: bool = False
    training_size: str = 'default'
    show_trace: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    min_lr_rate: Optional[float] = None
    mm_projector_lr: Optional[float] = None
    vision_tokenizer_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    local_run: bool = False
    max_grad_norm: float = 1.0
    

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        if getattr(trainer.args, "tune_vision_tokenizer", 'none') == "posembed":
            keys_to_match.extend(['posembed'])
        elif getattr(trainer.args, "tune_vision_tokenizer", 'none') == "decoder":
            keys_to_match.extend(['sem_seg_head.predictor'])
        elif getattr(trainer.args, "tune_vision_tokenizer", 'none') == "all":
            keys_to_match.extend(['vision_tower'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    if isinstance(special_tokens_dict, list):
        num_new_tokens = tokenizer.add_tokens(special_tokens_dict, special_tokens=True)
    else:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))    

    new_vocab_size = len(tokenizer)    
    # Update base model and current model config
    if hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    else:
        model.config.vocab_size = new_vocab_size
    model.vocab_size = new_vocab_size

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def make_supervised_data_module(processor: MagmaProcessor,
                                data_args, 
                                training_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = build_joint_dataset(
        processor=processor,
        data_path=data_args.data_path,
        data_args=data_args
    )

    if training_args.evaluation_strategy != 'no':
        val_dataset = build_joint_dataset(
            processor=processor,
            data_path=data_args.data_path,
            data_args=data_args,
            is_eval=True
        )    
    else:
        val_dataset = None
    data_collator = DataCollatorForSupervisedDataset(processor=processor)

    return dict(train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    if training_args.min_lr_rate is not None:
        training_args.lr_scheduler_kwargs = {'min_lr_rate': training_args.min_lr_rate}

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if 'magma' in model_args.model_name_or_path.lower():
        model = MagmaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2" if model_args.flash_attn_2_enabled else None,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
        magma_processor = MagmaProcessor.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )        
        model.config.tokenizer_vocab_size = magma_processor.tokenizer.vocab_size
    else:
        vision_config = {
            "img_size": model_args.img_size,
            "anyres_strategy": model_args.img_anyres_strategy,
            "vision_backbone": model_args.vision_backbone,
            "vision_tower": model_args.vision_tower,
            "vision_tower_ckpt": model_args.vision_tower_ckpt,
            "mm_vision_select_layer": model_args.mm_vision_select_layer,
            "mm_vision_select_feature": model_args.mm_vision_select_feature,
            "pretrain_mm_mlp_adapter": model_args.pretrain_mm_mlp_adapter,
            "mm_projector_type": model_args.mm_projector_type,
            "proj_vis_to_txt_tokens": model_args.proj_vis_to_txt_tokens,
            "mm_use_im_patch_token": model_args.mm_use_im_patch_token,
            "vision_feature_layer": "clip_vis_dense",
            "use_cache": False,
        }        
        text_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, 
            trust_remote_code=True
        )
        magma_config = MagmaConfig(
            vision_config=vision_config,
            text_config=text_config,
        )
        model = MagmaForCausalLM(magma_config)
        # reload language model
        model.language_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2" if model_args.flash_attn_2_enabled else None,      
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
            **bnb_model_from_pretrained_args
        )     
        # reload vision encoder
        from open_clip.pretrained import download_pretrained_from_hf
        if vision_config['vision_tower'] == 'convnext':
            model_id = 'laion/CLIP-convnext_large-laion2B-s34B-b82K-augreg'
        else:
            model_id = 'laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg'  
        checkpoint_path = download_pretrained_from_hf(model_id, cache_dir=None)
        model.load_special_module_from_ckpt(checkpoint_path, torch_dtype=(torch.bfloat16 if training_args.bf16 else None))

        # load 'magma/default_preprocessor_config.json' if it exists
        if os.path.exists('magma/default_preprocessor_config.json'):
            with open('magma/default_preprocessor_config.json') as f:
                preprocessor_config = json.load(f)
        else:
            preprocessor_config = {}
        image_processor = MagmaImageProcessor(**preprocessor_config)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        magma_processor = MagmaProcessor(image_processor=image_processor, tokenizer=tokenizer)

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=["<image>"],
            tokenizer=magma_processor.tokenizer,
            model=model,
        )

        # if tokenizer does not have pad_token, add it
        if magma_processor.tokenizer.pad_token_id is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict={'pad_token': '<pad>'},
                tokenizer=magma_processor.tokenizer,
                model=model,
            )

        model.config.image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        model.config.tokenizer_vocab_size = magma_processor.tokenizer.vocab_size
        
    model = model.to(training_args.device)
    
    magma_processor.tokenizer.model_max_length = training_args.model_max_length
    magma_processor.image_processor.base_img_size = model_args.img_size
    magma_processor.image_processor.anyres_strategy = model_args.img_anyres_strategy

    if model_args.mm_use_trace_start_end:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=["<trace_start>", "<trace_end>"],
            tokenizer=magma_processor.tokenizer,
            model=model,
        )

    if model_args.mm_use_image_start_end:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=["<image_start>", "<image_end>"],
            tokenizer=magma_processor.tokenizer,
            model=model,
        )

    # we add an <action> token as the place holder for the action
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=["<action>"],
        tokenizer=magma_processor.tokenizer,
        model=model,
    )

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = True

    if training_args.freeze_mm_mlp_adapter:
        for p in model.multi_modal_projector.parameters():
            p.requires_grad = False

    if model_args.tune_vision_tokenizer == "none":
        for name, p in model.vision_tower.named_parameters():
            p.requires_grad = False

    total_params = get_model_param_count(model, trainable_only=True)
    rank0_print(f"Total trainable parameters: {total_params}")
    
    if training_args.bits in [4, 8]:
        model.multi_modal_projector.to(dtype=compute_dtype, device=training_args.device)
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_args.mm_use_trace_start_end = model_args.mm_use_trace_start_end
    data_args.mm_use_trace_speed = model_args.mm_use_trace_speed
    data_args.mm_use_image_start_end = model_args.mm_use_image_start_end
    data_args.mm_use_image_history = model_args.mm_use_image_history
    data_args.mm_use_som_tom = model_args.mm_use_som_tom
    data_args.mm_use_som_tom_orig_img = model_args.mm_use_som_tom_orig_img
    data_args.remove_static_trace_pts = model_args.remove_static_trace_pts
    data_args.spatial_quant_size = model_args.spatial_quant_size
    data_args.version = model_args.version
    data_args.local_run = training_args.local_run
    data_args.task = model_args.task
    
    model.config.mm_use_trace_start_end = model_args.mm_use_trace_start_end
    model.config.mm_use_trace_speed = model_args.mm_use_trace_speed
    model.config.mm_use_image_start_end = model_args.mm_use_image_start_end
    model.config.mm_use_image_history = model_args.mm_use_image_history
    model.config.remove_static_trace_pts = model_args.remove_static_trace_pts
    model.config.mm_use_som_tom = model_args.mm_use_som_tom
    model.config.mm_use_som_tom_orig_img = model_args.mm_use_som_tom_orig_img
    model.config.spatial_quant_size = model_args.spatial_quant_size
    model.config.img_size = model_args.img_size
    model.config.use_cache = False    
    
    model.config.vision_config['img_anyres_strategy'] = model_args.img_anyres_strategy

    data_module = make_supervised_data_module(processor=magma_processor,
                                              data_args=data_args,
                                              training_args=training_args)
        
    trainer = MagmaTrainer(model=model,
                    tokenizer=magma_processor.tokenizer,
                    args=training_args,
                    **data_module)
    
    # print training_args
    rank0_print(training_args)
    rank0_print(model_args)
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Resuming from checkpoint...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    trainer.save_state()
    model.config.use_cache = True
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)
        # save image_processor config for rank 0
        if training_args.local_rank == 0 or training_args.local_rank == -1:        
            magma_processor.image_processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
