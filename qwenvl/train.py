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

from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer
import os

import transformers
import torch
import torch.distributed as dist

from qwenvl.data.base import BaseDataset

from .utils import PruneOldStateCallback, get_logger, print_trainable_parameters, print_trainable_parameters_visual, rank0_print

from .data import avail_datasets, SFTDataset, BenchmarkDataset
from .argument import *

from transformers import Trainer, Qwen2_5_VLForConditionalGeneration

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.set_num_threads(1)

logger = get_logger(__name__)


def set_model(model, model_args):
  """Do not change the order"""
  if model_args.tune_mm_llm:
    for n, p in model.language_model.named_parameters():
      p.requires_grad = True
    model.lm_head.requires_grad = True
  else:
    for n, p in model.langauge_model.named_parameters():
      p.requires_grad = False
    model.lm_head.requires_grad = False
    
  if model_args.tune_mm_vision:
    for n, p in model.visual.named_parameters():
      p.requires_grad = True
  else:
    for n, p in model.visual.named_parameters():
      p.requires_grad = False

  if model_args.tune_mm_mlp:
    for n, p in model.visual.merger.named_parameters():
      p.requires_grad = True
  else:
    for n, p in model.visual.merger.named_parameters():
      p.requires_grad = False

  return model


def set_processor(processor, proc_args: ProcessingArguments, data_args: DataArguments):
  tokenizer = processor.tokenizer
  img_processor = processor.image_processor
  vid_processor = processor.video_processor

  tokenizer.model_max_length = data_args.model_max_length

  img_processor.default_to_square = proc_args.default_to_square
  vid_processor.default_to_square = proc_args.default_to_square
  img_processor.max_pixels = proc_args.image_max_pixels
  vid_processor.max_pixels = proc_args.video_max_pixels
  img_processor.min_pixels = proc_args.image_min_pixels
  vid_processor.min_pixels = proc_args.video_min_pixels

  return processor


def make_data_module(
    processor,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
    for_training=True
):
  """Make dataset and collator for training or evaluation."""
  BaseDataset.num_proc = data_args.num_proc
  ds_name = data_args.dataset_use.replace('_', '-')
  ds_class = avail_datasets[ds_name]['ds_class']

  ds = ds_class(
      name=ds_name,
      processor=processor,
      proc_args=proc_args,
      data_args=data_args,
  )
  collate_fn = ds.collate_fn
  return {
      'train_dataset': ds if for_training else None,
      'eval_dataset': ds if not for_training else None,
      'data_collator': collate_fn,
  }


def rank0_make_data_module(*args, **kwargs):
  data_module = None
  if not dist.is_initialized():
    return make_data_module(*args, **kwargs)
  if dist.get_rank() == 0:
    data_module = make_data_module(*args, **kwargs)
  dist.barrier(device_ids=[dist.get_rank()])
  data_module = data_module or make_data_module(*args, **kwargs)
  dist.barrier(device_ids=[dist.get_rank()])
  ds = data_module['train_dataset'] or data_module['eval_dataset']
  if hasattr(ds, 'bin_pkl_path') and dist.get_rank() == 0:
    ds.bin_pkl_path.unlink(missing_ok=True)
  return data_module


def set_min_lr(training_args: TrainingArguments):
  """Set minimum learning rate if specified."""
  if training_args.lr_scheduler_type == "cosine_with_min_lr" and training_args.min_lr_ratio is not None:
    if training_args.lr_scheduler_kwargs is None:
      training_args.lr_scheduler_kwargs = {}
    training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_lr_ratio * \
        training_args.learning_rate
  return training_args


def train(attn_implementation="flash_attention_2"):
  if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    logger.info(f"Hello from Slurm! Rank {rank}/{world_size}")

  parser = transformers.HfArgumentParser((
      ModelArguments,
      DataArguments,
      TrainingArguments,
      ProcessingArguments,
  ))

  model_args, data_args, training_args, proc_args = parser.parse_args_into_dataclasses()
  training_args = set_min_lr(training_args)

  processor = transformers.AutoProcessor.from_pretrained(
      model_args.model_name_or_path,
      use_fast=True,
  )

  processor = set_processor(processor, proc_args, data_args)

  data_module = rank0_make_data_module(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
      for_training=True
  )

  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path,
      attn_implementation=attn_implementation,
      torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
  )
  model = set_model(model, model_args)
  print_trainable_parameters_visual(model.visual)
  print_trainable_parameters(model.model)

  if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

  trainer = Trainer(
      model=model,
      processing_class=processor,
      args=training_args,
      callbacks=[PruneOldStateCallback],
      **data_module
  )
  rank0_print(f"Trainer save_strategy: {trainer.args.save_strategy, trainer.args.save_steps, trainer.args.save_total_limit}")

  last_checkpoint = None
  if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

  if last_checkpoint is not None:
    rank0_print(
        f"Checkpoint detected. Resuming training from {last_checkpoint}")
  else:
    rank0_print(
        f"No checkpoint found at {training_args.output_dir}. Starting training from scratch.")

  # Trainer will always save final model https://huggingface.co/docs/transformers/v4.52.3/en/main_classes/trainer#transformers.TrainingArguments.save_strategy
  trainer.train(resume_from_checkpoint=last_checkpoint)

if __name__ == "__main__":
  torch.set_num_threads(1)
  train(attn_implementation="flash_attention_2")
