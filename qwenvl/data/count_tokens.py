import torch
from ..argument import *
from ..train import make_data_module, set_processor

from . import logger

if __name__ == "__main__":
  torch.set_num_threads(1)
  parser = transformers.HfArgumentParser((
      ModelArguments,
      DataArguments,
      ProcessingArguments,
  ))

  model_args, data_args, proc_args = parser.parse_args_into_dataclasses()
  logger.info("Counting tokens, not training.")
  processor = transformers.AutoProcessor.from_pretrained(
      model_args.model_name_or_path,
      use_fast=True,
  )
  processor = set_processor(processor, proc_args, data_args)

  data_module = make_data_module(
      processor=processor, data_args=data_args, proc_args=proc_args, for_training=True
  )