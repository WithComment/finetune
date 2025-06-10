from pathlib import Path
import os
import json
from typing import Callable
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import logging
import torch
import torch.distributed as dist
import transformers

from qwenvl.train.argument import DataArguments, ModelArguments, VisionArguments
from qwenvl.train.train_qwen import set_processor



from qwenvl.data import data_list, benchmark_classes
from qwenvl.data.benchmark import Benchmark

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
  handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def log_header(
    model_path: str,
    data_args: DataArguments,
    world_size: int,
    logger: logging.Logger,
):
  logger.info(f"Running on {world_size} GPUs")
  logger.info(f"benchmark: {data_args}")
  logger.info(f"Model: {model_path}")


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


def get_gpu_indices(
    benchmark: Benchmark,
    world_size: int,
    local_rank: int
) -> Benchmark:
  """Split the benchmark across multiple GPUs."""
  items_per_gpu = len(benchmark) // world_size
  start_idx = local_rank * items_per_gpu
  end_idx = start_idx + items_per_gpu
  return range(start_idx, end_idx)


def _infer(
    model,
    benchmark: Benchmark,
    gpu_indices: list[dict],
    collate_fn: callable,
    gen_config: dict,
    processor: AutoProcessor,
) -> list[dict]:
  result = []
  for idx in tqdm(gpu_indices, disable=torch.distributed.get_rank() != 0):
    item = benchmark.ds[idx]
    conv = benchmark.make_conversation(item)
    with torch.no_grad():
      output_ids = model.generate(
          **collate_fn(conv).to(model.device),
          **gen_config
      )

    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    pred = output.split("assistant")[-1].strip().strip("\n")
    item['model_output'] = pred
    result.append(benchmark.drop_non_json_fields(item))

  return result


def save_gpu_result(
    result: list[dict],
    rank: int,
    output_dir: Path,
) -> None:
  with open(output_dir / f'temp_{rank}.jsonl', 'w') as f:
    for item in result:
      f.write(json.dumps(item) + '\n')


def gather_result(
    world_size,
    output_dir: Path,
    logger: logging.Logger
):
  all_result = []
  # Gather result from all GPUs
  for rank in range(world_size):
    with open(output_dir / f"temp_{rank}.jsonl", 'r') as f:
      rank_result = [json.loads(line) for line in f]
      all_result.extend(rank_result)

  with open(output_dir / 'results.jsonl', 'w') as f:
    for item in all_result:
      f.write(json.dumps(item) + '\n')

  # Clean up temporary files
  for rank in range(world_size):
    temp_file = output_dir / f"temp_{rank}.jsonl"
    if temp_file.exists():
      temp_file.unlink()

  logger.info(f"Inference complete. result saved to {output_dir / 'results.jsonl'}")
  logger.info(f"Total processed samples: {len(all_result)}")

  return all_result


def eval_on_one_benchmark(
    model: Qwen2_5_VLForConditionalGeneration,
    world_size: int,
    local_rank: int,
    output_dir: Path,
    benchmark: Benchmark,
    collate_fn: Callable,
) -> None:
  if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
    
  gpu_result = _infer(
      model,
      benchmark,
      gpu_indices=get_gpu_indices(benchmark, world_size, local_rank),
      collate_fn=collate_fn,
      gen_config=benchmark.generation_config,
      processor=benchmark.processor,
  )
  save_gpu_result(gpu_result, local_rank, output_dir)

  dist.barrier()
  if local_rank == 0:
    final_result = gather_result(world_size, output_dir, logger)
  dist.barrier()
  

def make_data_module(processor, data_args, proc_args):
  """Make dataset and collator for supervised fine-tuning."""
  dataset_config = data_list(data_args.dataset_use.split(","))[0]
  benchmark_class = benchmark_classes[dataset_config['dataset_class']]
  benchmark = benchmark_class(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
      ds_key=dataset_config['ds_key'],
  )
  collate_fn = benchmark.make_model_input
  return {'benchmark': benchmark, 'collate_fn': collate_fn}


def main(
    model_path: str,
    data_args: DataArguments,
    proc_args: VisionArguments,
):
  """Run inference on the benchmark using Qwen2-VL."""
  dist.init_process_group(backend="nccl")

  world_size = dist.get_world_size()
  local_rank = dist.get_rank()
  device = f"cuda:{local_rank}"

  model, processor = load_pretrained_qwen(model_path, device)
  processor = set_processor(proc_args, processor)
  data_module = None
  if dist.get_rank() == 0:
    data_module = make_data_module(
        processor=processor, data_args=data_args, proc_args=proc_args
    )
    print(
        f"Data module created with {len(data_module['benchmark'])} samples.")
  dist.barrier()
  data_module = data_module or make_data_module(
      processor=processor, data_args=data_args, proc_args=proc_args
  )
  if local_rank == 0:
    log_header(model_path, data_args, world_size, logger)

  output_dir = data_module['benchmark'].dataset_dir.parent / "results" / model_path
  eval_on_one_benchmark(
      model,
      world_size=world_size,
      local_rank=local_rank,
      output_dir=output_dir,
      **data_module
  )
  dist.destroy_process_group()
  return 0


if __name__ == "__main__":
  parser = transformers.HfArgumentParser((
      ModelArguments,
      DataArguments,
      VisionArguments,
  ))
  model_args, data_args, proc_args = parser.parse_args_into_dataclasses()
  
  main(
      model_args.model_name_or_path,
      data_args,
      proc_args,
  )
