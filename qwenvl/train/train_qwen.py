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

import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import transformers
import torch
import pathlib
import os

from qwenvl.data import data_list
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    ProcessingArguments
)
from qwenvl.train.trainer import (
    replace_qwen2_vl_attention_class,
    print_trainable_parameters
)
from transformers import Trainer, Qwen2_5_VLForConditionalGeneration

from transformers.utils import logging as hf_logging

local_rank = None
hf_logging.set_verbosity_error()
torch.set_num_threads(1)


def rank0_print(*args):
  if local_rank == 0:
    print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
  """Collects the state dict and dump to disk."""

  if trainer.deepspeed:
    torch.cuda.synchronize()
    trainer.save_model(output_dir)
    return

  state_dict = trainer.model.state_dict()
  if trainer.args.should_save:
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict
    trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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


def train(attn_implementation="flash_attention_2"):
  global local_rank

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
  
  ds = data_list(data_args.dataset_use.split(","))[0]
  data_args.dataset_dir = ds['dataset_dir']
  data_args.media_dir = ds['media_dir']
  
  local_rank = training_args.local_rank
  data_module = None
  if transformers.trainer_utils.is_main_process(local_rank):
    # Some cpu and memory intensive preprocessing.
    data_module = make_supervised_data_module(
        processor=processor, data_args=data_args, proc_args=proc_args)
  torch.distributed.barrier()
  
  data_module = data_module or make_supervised_data_module(
      processor=processor, data_args=data_args, proc_args=proc_args
  )
  
  os.makedirs(training_args.output_dir, exist_ok=True)

  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path,
      cache_dir=training_args.cache_dir,
      attn_implementation=attn_implementation,
      torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
  )
  if torch.distributed.get_rank() == 0:
    model.visual.print_trainable_parameters()
    model.model.print_trainable_parameters()

  model.config.use_cache = False

  if training_args.gradient_checkpointing:
    if hasattr(model, "enable_input_require_grads"):
      model.enable_input_require_grads()
    else:
      def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
      model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
      
  trainer = Trainer(
      model=model, processing_class=processor, args=training_args, **data_module
  )
  
  if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    rank0_print("checkpoint found, resume training")
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()

  trainer.save_state()
  processor.save_pretrained(training_args.output_dir)

  model.config.use_cache = True

  safe_save_model_for_hf_trainer(
      trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
  torch.set_num_threads(1)
  if torch.cuda.get_device_name().split()[0] == 'Quadro':
    train(attn_implementation="eager")
  else:
    train(attn_implementation="flash_attention_2")
