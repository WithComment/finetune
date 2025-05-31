import base64
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
import argparse
import time
import torch
import os
import json
from tqdm import tqdm
from PIL import Image
import pandas as pd
import random
from datasets import load_dataset, Dataset
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader


logger = logging.getLogger(__name__)


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.common_utils import get_images_and_videos, get_video_frames

def setup_distributed():
  """Initialize distributed training."""
  dist.init_process_group(backend="nccl")
  torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def load_pretrained_qwen(model_name, model_path, device):
  """Load the Qwen model and processor."""
  if not model_path:
    model_path = model_name
  logger.info(f"Loading {model_path} on device {device}")
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    device_map={"": device},
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
  )
  processor = AutoProcessor.from_pretrained(model_name)
  return model, processor


def make_question(
    item: dict,
    processor: AutoProcessor,
    media_dir: str = None,
) -> tuple[list[dict], str]:
  """
  Return a formatted question that is suitable for the function `process_vision_info`.
  As well as the question after chat template has been applied.
  
  The formatted question will look like this:`
  [{
    "role": "user",
    "content": [
      {"type": "image", "image": "some_path.jpg"},
      {"type": "video", "video": "some_path.mp4"},
      {"type": "text", "text": "some more text"},
    ]
  }]
  `
  """
  q = item['question']
  content = list()
  for media in item.get('media', []):
    if not isinstance(media, str) or media.startswith('data:image;base64'):
      # Base64 or PIL.Image.Image
      content.append({"type": "image", "image": media})
    elif media.endswith('mp4'):
      content.append({"type": "video", "video": os.path.join(media_dir, media)})
    elif media.endswith(('jpg', 'jpeg', 'png')):
      content.append({"type": "image", "image": os.path.join(media_dir, media)})
  
  if not (options := item.get('options')):
    options = ['Y', 'N']
  
  q += "\nYou only options are:\n"
  for option in options:
    q += f"{option}\n"
    
  q += "Please answer with exactly one letter chosen from the options above."
  
  content.append({"type": "text", "text": q})
  question = [{
      "role": "user",
      "content": content
  }]
  # Add generation prompt appends "<|im_start|>assistant\n" to the text
  text = processor.apply_chat_template(
      question, tokenize=False, add_generation_prompt=True
  )
  return question, text


def load_benchmark(
    benchmark: str,
) -> tuple[Dataset, str, str]:
  with open('qwenvl/data/benchmarks.json', 'r') as f:
    benchmarks = json.load(f)
  benchmark = benchmarks[benchmark]
  ds_path = benchmark['dataset_path']
  if os.path.exists(ds_path):
    ds = Dataset.load_from_disk(ds_path)
  else:
    ds = load_dataset(ds_path)
  return ds, ds_path, benchmark['media_dir']


def get_gpu_dataset(
    dataset: Dataset,
    world_size: int,
    local_rank: int
) -> Dataset:
  """Split the dataset across multiple GPUs."""
  items_per_gpu = len(dataset) // world_size
  start_idx = local_rank * items_per_gpu
  end_idx = start_idx + items_per_gpu
  return dataset.select(range(start_idx, end_idx))


def log_header(
    logger: logging.Logger,
    world_size: int,
    benchmark: str,
    model: str,
    model_checkpoint: str,
    len_dataset: int,
    len_gpu_dataset: int
):
  logger.info(f"Running on {world_size} GPUs")
  logger.info(f"Dataset: {benchmark}")
  logger.info(f"Model: {model}")
  logger.info(f"Using model checkpoint: {model_checkpoint}")
  logger.info(f"Total samples: {len_dataset}")
  logger.info(f"Samples per GPU: ~{len_gpu_dataset}")  

  
def _infer(
    model,
    dataset: Dataset,
    local_rank: int,
    processor: AutoProcessor,
    media_dir: str,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
  result = []
  for i, item in enumerate(tqdm(dataset, disable=local_rank != 0)):
    item = deepcopy(item)
    question, text = make_question(item, processor, media_dir)
    image_inputs, video_inputs = get_images_and_videos(
        question, base_interval=None, min_frames=None, max_frames=None
    )
    
    image_inputs = image_inputs if image_inputs else None
    video_inputs = video_inputs if video_inputs else None
    
    inputs = processor(
      text=[text],
      images=image_inputs,
      videos=video_inputs,
      return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
      output_ids = model.generate(
          **inputs,
          max_new_tokens=max_new_tokens,
          do_sample=temperature > 0,
          temperature=temperature,
          use_cache=True
      )

    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    pred = output.split("assistant")[-1].strip().strip("\n")
    item['model_answer'] = pred
    item.pop('media', None)
    result.append(item)
    
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


def eval_acc(
    result: list[dict],
    output_dir: str,
    logger: logging.Logger
) -> tuple[float]:
  correct = 0
  invalid = 0
  total = len(result)
  for item in result:
    if len(item['model_answer']) != 1:
      invalid += 1
      continue
    if item['model_answer'].upper() == item['answer'].upper():
      correct += 1
  invalid_portion = invalid / total if total > 0 else 0
  acc = correct / (total - invalid) if total - invalid > 0 else 0
  with open(os.path.join(output_dir, 'acc.json'), 'w') as f:
    json.dump({
        'accuracy': acc,
        'invalid_portion': invalid_portion,
        'total': total,
        'correct': correct,
        'invalid': invalid
    }, f, indent=2)
  logger.info(f"Total: {total}, Correct: {correct}, Invalid: {invalid}")
  logger.info(f"Accuracy: {acc:.3f}, Invalid Portion: {invalid_portion:.3f}")
  return acc, invalid_portion, total, correct, invalid

def main(
    benchmark: str,
    model_name: str,
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    logger: logging.Logger,
):
  """Run inference on the dataset using Qwen2-VL."""
  # Setup distributed training
  setup_distributed()
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = dist.get_world_size()
  device = f"cuda:{local_rank}"
  
  if not model_path:
    model_path = model_name
  
  dataset, _, media_dir = load_benchmark(benchmark)
  
  if local_rank == 0:
    log_header(logger, world_size, benchmark, model_name, model_path,
               len(dataset), len(dataset) // world_size)
    
  model, processor = load_pretrained_qwen(model_name, model_path, device)
  gpu_dataset = get_gpu_dataset(dataset, world_size, local_rank)
  gpu_result = _infer(
      model,
      gpu_dataset,
      local_rank,
      processor,
      media_dir,
      max_new_tokens,
      temperature
  )
  output_dir = os.path.join(
    os.environ['BENCHMARKS_DIR'], benchmark, 'output', model_path.split('/')[-1])
  os.makedirs(output_dir, exist_ok=True)
  save_gpu_result(gpu_result, local_rank, output_dir)
  
  dist.barrier()
  if local_rank == 0:
    final_result = gather_result(world_size, output_dir, logger)
    eval_acc(final_result, output_dir, logger)
  dist.barrier()
  dist.destroy_process_group()
  return 0


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
  parser.add_argument("--model_path", type=str, required=False)
  parser.add_argument("--benchmark", type=str, default="vqa-rad/yes-no")
  parser.add_argument("--temperature", type=float, default=0.2)
  parser.add_argument("--max_new_tokens", type=float, default=8)
  args = parser.parse_args()
  
  logger = logging.getLogger(__name__)
  main(
      benchmark=args.benchmark,
      model_name=args.model_name,
      model_path=args.model_path,
      max_new_tokens=args.max_new_tokens,
      temperature=args.temperature,
      logger=logger
  )
