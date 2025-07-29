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

from pathlib import Path
import datasets
from transformers.trainer_utils import get_last_checkpoint
from transformers import Trainer
import os

import transformers
import torch
import torch.distributed as dist

from qwenvl import module, utils
from qwenvl.module import create_strategies, create_module

from .utils import PruneOldStateCallback, get_logger, print_trainable_parameters, print_trainable_parameters_visual, rank0_print

from .data import avail_datasets
from .argument import *

from transformers import Trainer, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration


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
  img_processor.do_resize = True

  return processor


def set_min_lr(training_args: TrainingArguments):
  """Set minimum learning rate if specified."""
  if training_args.lr_scheduler_type == "cosine_with_min_lr" and training_args.min_lr_ratio is not None:
    if training_args.lr_scheduler_kwargs is None:
      training_args.lr_scheduler_kwargs = {}
    training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_lr_ratio * \
        training_args.learning_rate
  return training_args

def create_datamodule(
    processor: transformers.AutoProcessor,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
):
  rank = dist.get_rank() if dist.is_initialized() else 0
  preprocess_strategies, cp, ip = create_strategies(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
      rank=rank,
  )
  
  ds, collate_fn = None, None
  if rank == 0:
    ds, collate_fn = create_module(
        data_args,
        preprocess_strategies=preprocess_strategies,
        cp=cp,
        ip=ip,
        rank=rank,
    )
  dist.barrier()
  if rank != 0:
    ds, collate_fn = create_module(
        data_args,
        preprocess_strategies=preprocess_strategies,
        cp=cp,
        ip=ip,
        rank=rank,
    )
  return ds, collate_fn

def train(attn_implementation="flash_attention_2"):
  
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
  
  training_args.deepspeed = "/projects/cft_vlm/finetune/qwenvl/scripts/zero3.json"

  custom_prompt = []
  if proc_args.cft_prompt:
    custom_prompt.append(f"cft_{proc_args.cft_prompt.replace(',', '_')}")
  if proc_args.sys_prompt:
    custom_prompt.append(f"sys_{proc_args.sys_prompt.replace(',', '_')}")
  if proc_args.usr_prompt:
    custom_prompt.append(f"usr_{proc_args.usr_prompt.replace(',', '_')}")
  
  output_dir = Path(training_args.output_dir)
  if custom_prompt:
    output_dir = output_dir.with_name("_".join([output_dir.name, "_".join(custom_prompt)]))
  
  global logger
  logger = get_logger(__name__, log_file=output_dir / 'training.log')
  module.set_logger(logger)
  utils.set_logger(logger)
  training_args.output_dir = str(output_dir)
  
  logger.info(f"Output file {training_args.output_dir}")

  processor = set_processor(processor, proc_args, data_args)

  ds, collate_fn = create_datamodule(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
  )
  
  model_class = Qwen2_5_VLForConditionalGeneration
    
  model = model_class.from_pretrained(
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
      train_dataset=ds,
      data_collator=collate_fn,
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

  trainer.train(resume_from_checkpoint=last_checkpoint)

  # When training completes normally without preemption.
  trainer.save_state()
  trainer.save_model(training_args.output_dir)
  
  if trainer.is_world_process_zero():
    processor.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
  torch.set_num_threads(1)
  train(attn_implementation="flash_attention_2")
