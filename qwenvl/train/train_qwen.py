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

from transformers import AutoProcessor, Trainer
from argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
)
from trainer import replace_qwen2_vl_attention_class
import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_qwen_packed import make_supervised_data_module_packed
local_rank = None


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
  """Set tuneable parameters for the model based on model_args."""
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


def set_processor(data_args, processor):
  """Set the processor parameters based on data_args."""
  # TODO: add shortest_edge if needed.
  img_processor = processor.image_processor
  vid_processor = processor.video_processor
  img_processor.max_pixels = data_args.max_pixels
  img_processor.min_pixels = data_args.min_pixels
  vid_processor.max_frame_pixels = data_args.video_max_frame_pixels
  vid_processor.min_frame_pixels = data_args.video_min_frame_pixels

def train(attn_implementation="flash_attention_2"):
  global local_rank

  parser = transformers.HfArgumentParser(
      (ModelArguments, DataArguments, TrainingArguments)
  )
  model_args, data_args, training_args = parser.parse_args_into_dataclasses()

  local_rank = training_args.local_rank
  os.makedirs(training_args.output_dir, exist_ok=True)

  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_args.model_name_or_path,
      cache_dir=training_args.cache_dir,
      attn_implementation=attn_implementation,
      torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
  )
  processor = AutoProcessor.from_pretrained(
      model_args.model_name_or_path)

  if data_args.data_flatten:
    replace_qwen2_vl_attention_class()
  model.config.use_cache = False

  if training_args.gradient_checkpointing:
    if hasattr(model, "enable_input_require_grads"):
      model.enable_input_require_grads()
    else:
      def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)

      model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

  tokenizer = transformers.AutoTokenizer.from_pretrained(
      model_args.model_name_or_path,
      cache_dir=training_args.cache_dir,
      model_max_length=training_args.model_max_length,
      padding_side="right",
      use_fast=False,
  )
  set_model(model_args, model)

  if torch.distributed.get_rank() == 0:
    model.visual.print_trainable_parameters()
    model.model.print_trainable_parameters()

  data_module = make_supervised_data_module_packed(
      processor, data_args=data_args)
  
  trainer = Trainer(
      model=model, processing_class=tokenizer, args=training_args, **data_module
  )

  if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    logging.info("checkpoint found, resume training")
    trainer.train(resume_from_checkpoint=True)
  else:
    trainer.train()
  trainer.save_state()
  processor.save_pretrained(training_args.output_dir)

  model.config.use_cache = True

  safe_save_model_for_hf_trainer(
      trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
  train(attn_implementation="flash_attention_2")
