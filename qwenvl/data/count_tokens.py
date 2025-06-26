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
  ds = data_module['train_dataset']
  ds_bin_lengths = []
  for bin in ds:
    num_tokens = sum([item['num_tokens'] for item in bin])
    ds_bin_lengths.append(num_tokens)
  logger.info(f"Total number of bins: {len(ds_bin_lengths)}")
  logger.info(f"Total number of tokens in bins: {sum(ds_bin_lengths)}")
  logger.info(f"Average number of tokens per bin: {sum(ds_bin_lengths) / len(ds_bin_lengths):.2f}")
  logger.info(f"Max number of tokens in a bin: {max(ds_bin_lengths)}, idx: {ds_bin_lengths.index(max(ds_bin_lengths))}")
  logger.info(f"Min number of tokens in a bin: {min(ds_bin_lengths)}, idx: {ds_bin_lengths.index(min(ds_bin_lengths))}")
