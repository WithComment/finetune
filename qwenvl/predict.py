from pathlib import Path
import os
import json
from typing import Callable
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
import logging
import torch
import torch.distributed as dist
import transformers

from qwenvl.eval import comp_answer_basic, evaluate, yes_no_filter

from .argument import DataArguments, ModelArguments, ProcessingArguments
from .train import make_data_module, rank0_make_data_module, set_processor
from .utils import get_logger
from .data import (
    BenchmarkDataset
)
logger = get_logger(__name__)


def log_header(
    model_path: str,
    data_args: DataArguments,
    world_size: int,
    logger: logging.Logger,
):
  logger.info(f"Running on {world_size} GPUs")
  logger.info(f"benchmark: {data_args}")
  logger.info(f"Model: {model_path}")


def _load_model_and_processor(
    model_path: str,
    device: str,
) -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
  """Load the Qwen model and processor."""
  logger.info(f"Loading model from {model_path} on device {device}")
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      model_path,
      device_map={"": device},
      torch_dtype=torch.float16,
      attn_implementation="flash_attention_2"
  )
  processor = AutoProcessor.from_pretrained(model_path)
  return model, processor


def load_pretrained_qwen(model_path, device):
  """Load the Qwen model and processor."""
  checkpoint_dir = Path(os.environ.get('CHECKPOINT_DIR', ''))
  if (checkpoint_dir / model_path).exists():
    model_path = checkpoint_dir / model_path

  try:
    model, processor = _load_model_and_processor(model_path, device)
  except OSError as e:
    logger.warning(
        f"Model not found at {model_path}. Attempting to load checkpoint.")
    model_path = get_last_checkpoint(model_path)
    if model_path is None:
      raise RuntimeError(f"No checkpoint found in {model_path}.") from e
    model, processor = _load_model_and_processor(model_path, device)

  return model, processor, Path(model_path)


def get_gpu_indices(
    benchmark: BenchmarkDataset,
    world_size: int,
    local_rank: int
) -> range:
  """Split the benchmark across multiple GPUs."""
  items_per_gpu = len(benchmark) // world_size
  start_idx = local_rank * items_per_gpu
  end_idx = start_idx + items_per_gpu
  return range(start_idx, end_idx)


def _infer(
    model,
    benchmark: BenchmarkDataset,
    gpu_indices: list[dict],
    collate_fn: callable,
    gen_config: dict,
    processor: AutoProcessor,
) -> list[dict]:
  result = []
  for idx in tqdm(gpu_indices, disable=torch.distributed.get_rank() != 0):
    item = benchmark[idx]
    with torch.no_grad():
      output_ids = model.generate(
          **collate_fn(item).to(model.device),
          **gen_config
      )

    outputs = processor.batch_decode(output_ids, skip_special_tokens=True)
    for item, output in zip(item, outputs):
      output = output.split("assistant")[-1].strip().strip("\n")
      item['model_answer'] = output
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

  logger.info(
      f"Inference complete. result saved to {output_dir / 'results.jsonl'}")
  logger.info(f"Total processed samples: {len(all_result)}")

  return all_result


def generate_output(
    model: Qwen2_5_VLForConditionalGeneration,
    world_size: int,
    local_rank: int,
    output_dir: Path,
    benchmark: BenchmarkDataset,
    collate_fn: Callable,
    gen_config: dict,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  gpu_result = _infer(
      model,
      benchmark,
      gpu_indices=get_gpu_indices(benchmark, world_size, local_rank),
      collate_fn=collate_fn,
      gen_config=gen_config,
      processor=benchmark.processor,
  )
  save_gpu_result(gpu_result, local_rank, output_dir)

  dist.barrier()
  if local_rank == 0:
    final_result = gather_result(world_size, output_dir)
  dist.barrier()


def predict(
    model_path: str,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
):
  """Run inference on the benchmark using Qwen2-VL."""
  dist.init_process_group(backend="nccl")

  world_size = dist.get_world_size()
  local_rank = dist.get_rank()
  device = f"cuda:{local_rank}"

  model, processor, _ = load_pretrained_qwen(model_path, device)
  processor = set_processor(processor, proc_args, data_args)
  data_module = rank0_make_data_module(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
      for_training=False
  )
  if local_rank == 0:
    log_header(model_path, data_args, world_size, logger)

  eval_dataset = data_module['eval_dataset']
  collate_fn = data_module['data_collator']

  output_dir = eval_dataset.ds_dir.parent.parent / 'results' / data_args.split / Path(model_path).name
  generate_output(
      model,
      world_size=world_size,
      local_rank=local_rank,
      output_dir=output_dir,
      benchmark=eval_dataset,
      collate_fn=collate_fn,
      gen_config={
          'max_new_tokens': 32,
          'do_sample': False,
      }
  )
  return output_dir


if __name__ == "__main__":
  parser = transformers.HfArgumentParser((
      ModelArguments,
      DataArguments,
      ProcessingArguments,
  ))
  model_args, data_args, proc_args = parser.parse_args_into_dataclasses()

  output_dir = predict(
      model_args.model_name_or_path,
      data_args,
      proc_args,
  )
  if dist.get_rank() == 0:
    summary = evaluate(
        output_dir / 'results.jsonl',
        comp_answer=comp_answer_basic,
        filter=yes_no_filter
    )
    with open(output_dir / 'summary.json', 'w') as f:
      json.dump(summary, f, indent=2)
    logger.info(f"Evaluation summary saved to {output_dir / 'summary.json'}")
    logger.info(summary)
  dist.barrier(device_ids=[dist.get_rank()])
  dist.destroy_process_group()
