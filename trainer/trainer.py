import os
from typing import List, Optional

import torch
from torch.cuda import synchronize
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
    logger,
)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    # indices = torch.arange(len(lengths))
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size

        # gather self lengths from all processes
        # if self.world_size > 1:
        #     # gather the size of lengths from all processes
        #     sizes = torch.tensor([len(lengths)], device=torch.device("cuda"))
        #     # take minimum length
        #     torch.distributed.all_reduce(sizes, op=torch.distributed.ReduceOp.MIN)
        #     min_size = sizes.item()
        #     # trim lengths to the minimum size
        #     lengths = lengths[:min_size]

        #     # append lengths from all processes
        #     all_lengths = [torch.zeros_like(lengths) for _ in range(self.world_size)]
        #     torch.distributed.all_gather(all_lengths, torch.tensor(lengths, device=torch.device("cuda")))
        #     lengths = torch.cat(all_lengths, dim=0).tolist()
        #     if torch.distributed.get_rank() == 0:
        #         import pdb; pdb.set_trace()

        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class MagmaTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_moving_average = None

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def evaluation_loop(self, dataloader, description: str, prediction_loss_only: Optional[bool] = None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Override the `evaluation_loop` method for custom evaluation.
        """
        # Custom logic before evaluation loop starts
        print(f"Starting custom evaluation loop: {description}")
        pass
    
        synchronize()  # This ensures all GPU operations are completed        
        # Initialize containers for predictions and labels
        all_preds = []
        all_labels = []
        # Iterate over the evaluation data loader
        for step, inputs in enumerate(dataloader):
            # Optionally, apply the data collator manually here if needed
            # inputs = self.data_collator(inputs)

            # Move batch to the appropriate device
            inputs = self._prepare_inputs(inputs)

            # Disable gradient calculation during evaluation
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Extract logits (predictions) and labels
            logits = outputs.logits
            labels = inputs['labels']

            # Collect predictions and labels for this batch
            preds = logits.argmax(dim=-1)  # Assuming classification task
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Concatenate all batches to get complete predictions and labels
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Custom metric computation logic can be applied here
        metrics = self.compute_metrics((all_preds, all_labels))

        print(f"Finished custom evaluation loop. Metrics: {metrics}")
        
        # Return the results as expected by the trainer (you can customize this)
        return metrics

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = []
            for n, p in opt_model.named_parameters():
                if p.requires_grad:
                    if "mm_projector" in n and self.args.mm_projector_lr is not None:
                        optimizer_grouped_parameters.append(
                            {
                                "params": [p],
                                "weight_decay": self.args.weight_decay if n in decay_parameters else 0.0,
                                "lr": self.args.mm_projector_lr,
                            }
                        )
                    elif "vision_tower" in n and self.args.vision_tokenizer_lr is not None:
                        optimizer_grouped_parameters.append(
                            {
                                "params": [p],
                                "weight_decay": self.args.weight_decay if n in decay_parameters else 0.0,
                                "lr": self.args.vision_tokenizer_lr,
                            }
                        )
                    else:
                        optimizer_grouped_parameters.append(
                            {
                                "params": [p],
                                "weight_decay": self.args.weight_decay if n in decay_parameters else 0.0,
                            }
                        )
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")
        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler', 'segtok_']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            if getattr(self.args, "tune_vision_tokenizer", 'none') == "posembed":
                keys_to_match.extend(['posembed'])
            elif getattr(self.args, "tune_vision_tokenizer", 'none') == "decoder":
                keys_to_match.extend(['sem_seg_head.predictor'])
            elif getattr(self.args, "tune_vision_tokenizer", 'none') == "all":
                keys_to_match.extend(['vision_tower'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                print(f"keys to match: {keys_to_match}")            
                print(f"save checkpoint to {os.path.join(output_dir, 'mm_projector.bin')}")
                torch.save(weight_to_save, os.path.join(output_dir, 'mm_projector.bin'))
        else:
            super(MagmaTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(MagmaTrainer, self)._save(output_dir, state_dict)
