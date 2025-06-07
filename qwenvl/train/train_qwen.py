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
from transformers.trainer_callback import TrainerCallback, TrainerState, TrainerControl
from transformers import Trainer
import logging
import signal
import os
import sys
from pathlib import Path

import transformers
import torch
import torch.distributed as dist
import pathlib

from qwenvl.data import data_list, dataset_classes
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    ProcessingArguments
)
import qwenvl.train.trainer

from transformers import Trainer, Qwen2_5_VLForConditionalGeneration

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.set_num_threads(1)


def rank0_print(*args):
  if dist.get_rank() == 0:
    print(*args)


def set_model(model_args, model):
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

  if model_args.tune_mm_llm:
    for n, p in model.model.named_parameters():
      p.requires_grad = True
    model.lm_head.requires_grad = True
  else:
    for n, p in model.model.named_parameters():
      p.requires_grad = False
    model.lm_head.requires_grad = False


def set_processor(processor_args, processor):
  tokenizer = processor.tokenizer
  img_processor = processor.image_processor
  vid_processor = processor.video_processor

  tokenizer.padding_side = processor_args.padding_side
  tokenizer.model_max_length = processor_args.model_max_length

  img_processor.max_pixels = processor_args.image_max_pixels
  img_processor.min_pixels = processor_args.image_min_pixels
  img_processor.size["shortest_edge"] = processor_args.shortest_edge

  vid_processor.min_pixels = processor_args.video_min_pixels
  vid_processor.max_pixels = processor_args.video_max_pixels
  vid_processor.min_frame_pixels = processor_args.video_min_pixels
  vid_processor.max_frame_pixels = processor_args.video_max_pixels
  vid_processor.size['shortest_edge'] = processor_args.shortest_edge
  vid_processor.default_to_square = processor_args.video_default_to_square

  return processor


def make_data_module(processor, data_args, proc_args):
  """Make dataset and collator for supervised fine-tuning."""
  dataset_config = data_list(data_args.dataset_use.split(","))[0]
  dataset_class = dataset_classes[dataset_config['dataset_class']]
  train_dataset = dataset_class(
      processor=processor,
      proc_args=proc_args,
      data_args=data_args,
      sampling_rate=dataset_config['sampling_rate']
  )
  collate_fn = train_dataset.make_model_input
  return dict(
      train_dataset=train_dataset, eval_dataset=None, data_collator=collate_fn
  )


SLURM_PREEMPTION_SIGNAL_RECEIVED = False

def handle_preemption_signal(signum, frame):
  """Signal handler that sets the global preemption flag."""
  global SLURM_PREEMPTION_SIGNAL_RECEIVED
  if not SLURM_PREEMPTION_SIGNAL_RECEIVED:
    rank0_print(
        f"Received signal {signum}. Triggering graceful shutdown and checkpointing...")
    SLURM_PREEMPTION_SIGNAL_RECEIVED = True

class SlurmPreemptionCallback(TrainerCallback):
  def on_step_end(self, args: transformers.TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    """Check for the preemption signal at the end of each step."""
    if SLURM_PREEMPTION_SIGNAL_RECEIVED:
      if state.is_world_process_zero:
        rank0_print("Preemption signal detected. Saving model...")

      kwargs['trainer'].save_model(args.output_dir)
      kwargs['trainer'].save_state()

      control.should_training_stop = True

    dist.barrier()


def train(attn_implementation="flash_attention_2"):
  if "SLURM_PROCID" in os.environ:
    dist.init_process_group(backend='nccl')
  
  if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(
        f"Hello from Slurm! Rank {rank}/{world_size}",
        flush=True)

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

  processor = set_processor(proc_args, processor)

  data_module = None
  if dist.get_rank() == 0:
    data_module = make_data_module(
        processor=processor, data_args=data_args, proc_args=proc_args
    )
    rank0_print(
        f"Data module created with {len(data_module['train_dataset'])} training samples.")
  dist.barrier()
  data_module = data_module or make_data_module(
      processor=processor, data_args=data_args, proc_args=proc_args
  )

  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path,
      attn_implementation=attn_implementation,
      torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
  )
  if dist.get_rank() == 0:
    model.visual.print_trainable_parameters()
    model.model.print_trainable_parameters()

  if training_args.gradient_checkpointing:
    if hasattr(model, "enable_input_require_grads"):
      model.enable_input_require_grads()
    else:
      def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
      model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

  trainer = Trainer(
      model=model,
      processing_class=processor,
      args=training_args,
      callbacks=[SlurmPreemptionCallback],
      **data_module
  )

  last_checkpoint = None
  if os.path.isdir(training_args.output_dir):
    last_checkpoint = get_last_checkpoint(training_args.output_dir)

  if last_checkpoint is not None:
    rank0_print(
        f"Checkpoint detected. Resuming training from {last_checkpoint}")
  else:
    rank0_print("No checkpoint found. Starting training from scratch.")
  
  trainer.train(resume_from_checkpoint=last_checkpoint)

  # When training completes normally without preemption.
  trainer.save_state()
  if trainer.is_world_process_zero():
    processor.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
  torch.set_num_threads(1)
  train(attn_implementation="flash_attention_2")
