from pathlib import Path
import os
import json
from typing import Callable
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
import logging
import torch
import torch.distributed as dist
import transformers

from qwenvl.eval import comp_answer_basic, evaluate, yes_no_filter

from .argument import DataArguments, ModelArguments, ProcessingArguments
from .train import rank0_print, set_processor, create_datamodule
from .utils import get_logger
from .data import avail_datasets
from .data.module import DatasetWrapper

logger = get_logger(__name__)
transformers.logging.set_verbosity_error()

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
) -> tuple[AutoModel, AutoProcessor]:
  """Load the Qwen model and processor."""
  if not isinstance(model_path, str):
    model_path = str(model_path)
  logger.info(f"Loading model from {model_path} on device {device}")
  if 'Qwen2-VL' in model_path:
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map={"": device},
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
  elif 'Qwen2.5-VL' in model_path:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map={"": device},
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2"
    )
  else:
    raise ValueError(f"Unsupported model type: {model_path}")
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
    benchmark: DatasetWrapper,
    world_size: int,
    local_rank: int,
    portion: float,
) -> range:
  """Split the benchmark across multiple GPUs."""
  total = int(len(benchmark) * portion)
  items_per_gpu = total // world_size
  start_idx = local_rank * items_per_gpu
  end_idx = start_idx + items_per_gpu
  return range(start_idx, end_idx)

def drop_non_json_fields(item: dict) -> dict:
  """Drop fields that are not JSON serializable."""
  return {k: v for k, v in item.items() if isinstance(v, (str, int, float, bool, list, dict))}

def _infer(
    model,
    benchmark: DatasetWrapper,
    gpu_indices: list[dict],
    collate_fn: callable,
    gen_config: dict,
    processor: AutoProcessor,
) -> list[dict]:
  result = []
  # logger.info(collate_fn([benchmark[gpu_indices[0]]]))
  for idx in tqdm(gpu_indices, disable=torch.distributed.get_rank() != 0):
    # Turn into a batch of size 1
    batch = [benchmark[idx]]
    with torch.inference_mode():
      input = collate_fn(batch)
      input.pop('labels', None)
      input = input.to(model.device)
      output_ids = model.generate(
          **input,
          **gen_config
      )

    outputs = processor.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    for item, output in zip(batch, outputs):
      output = output.split("assistant")[-1].strip().strip("\n")
      # Unpack item.
      item = item[0]
      item['model_answer'] = output
      result.append(drop_non_json_fields(item))
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
    model: AutoModel,
    world_size: int,
    local_rank: int,
    output_dir: Path,
    benchmark: DatasetWrapper,
    processor: AutoProcessor,
    collate_fn: Callable,
    gen_config: dict,
    portion: float,
) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  gpu_result = _infer(
      model,
      benchmark,
      gpu_indices=get_gpu_indices(benchmark, world_size, local_rank, portion),
      collate_fn=collate_fn,
      gen_config=gen_config,
      processor=processor,
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

  model, processor, model_path = load_pretrained_qwen(model_path, device)
  processor = set_processor(processor, proc_args, data_args)
  ds, collate_fn = create_datamodule(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
  )
  if local_rank == 0:
    log_header(model_path, data_args, world_size, logger)
  
  if model_path.name.startswith('checkpoint-'):
    checkpoint_name = '-'.join([model_path.parts[-2], model_path.name.split('-')[-1]])
  else:
    checkpoint_name = model_path.name
    
  ds_dir = Path(avail_datasets[data_args.dataset_use]['ds_dir'])
  output_dir = ds_dir.parent.parent / 'results' / data_args.split / checkpoint_name
  logger.info(f"Output directory: {output_dir}")
  rank0_print(data_args)
  generate_output(
      model,
      world_size=world_size,
      local_rank=local_rank,
      output_dir=output_dir,
      benchmark=ds,
      collate_fn=collate_fn,
      processor=processor,
      gen_config={
          'max_new_tokens': 64,
          "eos_token_id": [151645, 151643],
          'do_sample': False,
          'repetition_penalty': 1.1,
      },
      portion=data_args.portion,
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
