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
import logging
import signal
import os

import transformers
import torch
import torch.distributed as dist

from qwenvl.data.base import BaseDataset
from qwenvl.data.openbiomedvid import OpenbiomedvidDataset

from .utils import get_logger

from .data import avail_datasets, SFTDataset, BenchmarkDataset
from .argument import *

from transformers import Trainer, Qwen2_5_VLForConditionalGeneration

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.set_num_threads(1)


logger = get_logger(__name__)


def rank0_print(msg, lvl="INFO"):
  lvl = getattr(logging, lvl.upper(), logging.INFO)
  if dist.get_rank() == 0:
    logger.log(level=lvl, msg=msg)


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

  img_processor.max_pixels = proc_args.image_max_pixels
  img_processor.min_pixels = proc_args.image_min_pixels
  img_processor.size["shortest_edge"] = proc_args.shortest_edge

  vid_processor.min_pixels = proc_args.video_min_pixels
  vid_processor.max_pixels = proc_args.video_max_pixels
  vid_processor.min_frame_pixels = proc_args.video_min_pixels
  vid_processor.max_frame_pixels = proc_args.video_max_pixels
  vid_processor.size['shortest_edge'] = proc_args.shortest_edge
  vid_processor.default_to_square = proc_args.video_default_to_square

  return processor


def make_data_module(
    processor,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
    for_training=True
):
  """Make dataset and collator for training or evaluation."""
  BaseDataset.num_proc = data_args.num_proc
  ds_name = data_args.dataset_use
  ds_class = avail_datasets[ds_name]['ds_class']
  
  if for_training:
    assert issubclass(ds_class, SFTDataset)
  else:
    assert issubclass(ds_class, BenchmarkDataset)
    
  if issubclass(ds_class, OpenbiomedvidDataset) and data_args.data_packing:
    logger.warning("OpenbiomedvidDataset does not support data packing due to video handling. Setting data_packing to False.")
    data_args.data_packing = False
    
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

def print_trainable_parameters_visual(model) -> None:
  """
  Prints the trainable status of all vision components including attention blocks and merger module.
  Outputs the indices of trainable/non-trainable blocks and the merger module status.
  """
  trainable_blocks = []
  non_trainable_blocks = []

  # Check trainable status of vision attention blocks
  for block_idx, block in enumerate(model.blocks):
    is_trainable = all(param.requires_grad for param in block.parameters())
    if is_trainable:
      trainable_blocks.append(block_idx)
    else:
      non_trainable_blocks.append(block_idx)

  # Check trainable status of merger module
  is_merger_trainable = any(
      param.requires_grad for param in model.merger.parameters())

  # Print results
  print("Vision Module - Attention Blocks:")
  print(
      f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
  )
  print(
      f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
  )
  print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(model) -> None:
  """
  Prints the trainable status of all LLM components including embeddings, layers, and normalization.
  Outputs the indices of trainable/non-trainable layers and other module statuses.
  """
  # Check embed_tokens
  model = model.language_model
  is_embed_trainable = any(
      param.requires_grad for param in model.embed_tokens.parameters()
  )
  print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

  # Check each decoder layer
  trainable_layers = []
  non_trainable_layers = []

  for layer_idx, layer in enumerate(model.layers):
    is_trainable = any(param.requires_grad for param in layer.parameters())
    if is_trainable:
      trainable_layers.append(layer_idx)
    else:
      non_trainable_layers.append(layer_idx)

  # Print layer status
  print(
      f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
  )
  print(
      f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
  )



def train(attn_implementation="flash_attention_2"):
  if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rank0_print(
        f"Hello from Slurm! Rank {rank}/{world_size}")

  signal.signal(signal.SIGUSR1, handle_preemption_signal)

  parser = transformers.HfArgumentParser((
      ModelArguments,
      DataArguments,
      TrainingArguments,
      ProcessingArguments,
  ))

  model_args, data_args, training_args, proc_args = parser.parse_args_into_dataclasses()

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
  if dist.get_rank() == 0:
    print_trainable_parameters_visual(model.visual)
    print_trainable_parameters(model.model)

  if training_args.gradient_checkpointing:
    model.enable_input_require_grads()

  trainer = Trainer(
      model=model,
      processing_class=processor,
      args=training_args,
      **data_module
  )

  last_checkpoint = None
  if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

  if last_checkpoint is not None:
    rank0_print(
        f"Checkpoint detected. Resuming training from {last_checkpoint}")
  else:
    rank0_print(f"No checkpoint found at {training_args.output_dir}. Starting training from scratch.")
  
  trainer.train(resume_from_checkpoint=last_checkpoint)

  # When training completes normally without preemption.
  trainer.save_state()
  if trainer.is_world_process_zero():
    processor.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
  torch.set_num_threads(1)
  train(attn_implementation="flash_attention_2")
