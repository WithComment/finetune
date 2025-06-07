import torch
from qwenvl.train.argument import *
from qwenvl.train.train_qwen import make_data_module, set_processor


if __name__ == "__main__":
  torch.set_num_threads(1)
  parser = transformers.HfArgumentParser((
      ModelArguments,
      DataArguments,
      ProcessingArguments,
  ))

  model_args, data_args, proc_args = parser.parse_args_into_dataclasses()
  if data_args.count_tokens:
    print("Counting tokens, not training.")
    processor = transformers.AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    processor = set_processor(proc_args, processor)

    data_module = make_data_module(
        processor=processor, data_args=data_args, proc_args=proc_args
    )