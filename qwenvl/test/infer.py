from copy import deepcopy
from pathlib import Path
import sys
import argparse
import time
import os
import json
from tqdm import tqdm
from datasets import datasets
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
import logging
import torch
import torch.distributed as dist

from qwenvl.train.argument import DataArguments, ProcessingArguments
from qwenvl.train.train_qwen import set_processor


logger = logging.getLogger(__name__)


from qwenvl.data import data_list, benchmark_classes
from qwenvl.data.benchmark import Benchmark

def setup_distributed():
  """Initialize distributed training."""
  dist.init_process_group(backend="nccl")
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_pretrained_qwen(model_path, device):
  """Load the Qwen model and processor."""
  logger.info(f"Loading {model_path} on device {device}")
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_path,
      device_map={"": device},
      torch_dtype=torch.float16,
      attn_implementation="flash_attention_2"
  )
  processor = AutoProcessor.from_pretrained(model_path)
  return model, processor


def get_gpu_benchmark(
    benchmark: Benchmark,
    world_size: int,
    local_rank: int
) -> Benchmark:
  """Split the benchmark across multiple GPUs."""
  items_per_gpu = len(benchmark) // world_size
  start_idx = local_rank * items_per_gpu
  end_idx = start_idx + items_per_gpu
  return benchmark[start_idx:end_idx]


def log_header(
    logger: logging.Logger,
    world_size: int,
    benchmark: str,
    model_path: str,
    len_benchmark: int,
    len_gpu_benchmark: int
):
  logger.info(f"Running on {world_size} GPUs")
  logger.info(f"benchmark: {benchmark}")
  logger.info(f"Model: {model_path}")
  logger.info(f"Total samples: {len_benchmark}")
  logger.info(f"Samples per GPU: ~{len_gpu_benchmark}")


def _infer(
    model,
    conversations: list[dict],
    collate_fn: callable,
    gen_config: dict,
    processor: AutoProcessor,
) -> list[dict]:
  result = []
  for i, item in tqdm(enumerate(conversations, disable=torch.distributed.get_rank() != 0)):

    with torch.no_grad():
      output_ids = model.generate(
          **collate_fn(item),
          **gen_config
      )

    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    pred = output.split("assistant")[-1].strip().strip("\n")
    result.append(pred)

  return result


def save_gpu_result(
    result: list[dict],
    rank: int,
    output_dir: str,
) -> None:
  with open(os.path.join(output_dir, f"temp_{rank}.jsonl"), 'w') as f:
    for item in result:
      f.write(json.dumps(item) + '\n')


def gather_result(
    world_size,
    output_dir: str,
    logger: logging.Logger
):
  all_result = []
  # Gather result from all GPUs
  for rank in range(world_size):
    rank_file = os.path.join(output_dir, f"temp_{rank}.jsonl")
    with open(rank_file, 'r') as f:
      rank_result = [json.loads(line) for line in f]
      all_result.extend(rank_result)

  all_result.sort(key=lambda x: x['id'])
  final_output = os.path.join(output_dir, "result.jsonl")
  with open(final_output, 'w') as f:
    for item in all_result:
      f.write(json.dumps(item) + '\n')

  # Clean up temporary files
  for rank in range(world_size):
    temp_file = os.path.join(output_dir, f"temp_{rank}.jsonl")
    if os.path.exists(temp_file):
      os.remove(temp_file)

  logger.info(f"Inference complete. result saved to {final_output}")
  logger.info(f"Total processed samples: {len(all_result)}")

  return all_result


def eval_on_one_benchmark(
    model: Qwen2_5_VLForConditionalGeneration,
    dataloader: torch.utils.data.DataLoader,
    world_size: int,
    local_rank: int,
) -> None:
  gpu_benchmark = get_gpu_benchmark(dataloader, world_size, local_rank)
  gpu_result = _infer(
      model,
      gpu_benchmark,
      local_rank,
      processor,
      media_dir,
      max_new_tokens,
      temperature
  )
  save_gpu_result(gpu_result, local_rank, output_dir)

  dist.barrier()
  if local_rank == 0:
    final_result = gather_result(world_size, output_dir, logger)
    eval_acc(final_result, output_dir, logger)
  dist.barrier()


def make_dataloader(processor, benchmark, proc_args):
  """Make dataset and collator for supervised fine-tuning."""
  dataset_config = data_list[benchmark]
  benchmark_class = benchmark_classes[dataset_config['dataset_class']]
  benchmark = benchmark_class(
      benchmark=benchmark,
      processor=processor,
      proc_args=proc_args,
      sampling_rate=dataset_config['sampling_rate']
  )
  collate_fn = benchmark.make_model_input
  return torch.utils.data.DataLoader(
      benchmark,
      collate_fn=collate_fn,
  )


def main(
    model_path: str,
    benchmark: str,
    split: str,
    proc_args: ProcessingArguments,
    logger: logging.Logger,
):
  """Run inference on the benchmark using Qwen2-VL."""
  setup_distributed()
  dist.init_process_group(backend="nccl")

  world_size = dist.get_world_size()
  local_rank = dist.get_rank()
  device = f"cuda:{local_rank}"

  model, processor = load_pretrained_qwen(model_path, device)
  processor = set_processor(processor, proc_args)
  if dist.get_rank() == 0:
    dataloader = make_dataloader(
        processor=processor, benchmark=benchmark, proc_args=proc_args
    )
    print(
        f"Data module created with {len(dataloader['train_dataset'])} training samples.")
  dist.barrier()
  dataloader = dataloader or make_dataloader(
      processor=processor, benchmark=benchmark, proc_args=proc_args
  )
  if local_rank == 0:
    log_header(logger, world_size, benchmark, model_path,
                len(benchmark), len(benchmark) // world_size)

  eval_on_one_benchmark(
      model,
      dataloader,
      world_size,
      local_rank
  )
  dist.destroy_process_group()
  return 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_path", type=str,
                      default="Qwen/Qwen2.5-VL-3B-Instruct")
  parser.add_argument("--benchmarks", type=str, default="vqa-rad/yes-no")
  parser.add_argument("--temperature", type=float, default=0.2)
  parser.add_argument("--max_new_tokens", type=float, default=8)
  args = parser.parse_args()

  logger = logging.getLogger(__name__)
  main(
      benchmarks=args.benchmarks,
      model_path=args.model_path,
      max_new_tokens=args.max_new_tokens,
      temperature=args.temperature,
      logger=logger
  )
