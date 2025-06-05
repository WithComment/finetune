import copy
from functools import partial
import json
import os
from pathlib import Path
import random
from typing import Callable
import datasets
from decord import VideoReader
import numpy as np
from PIL import Image

import torch

from transformers import AutoProcessor, PreTrainedTokenizer

import base64
from io import BytesIO

from qwenvl.train.argument import ProcessingArguments

PLACEHOLDER_IDS = {151654, 151655, 151656}

def read_jsonl(path):
  with open(path, "r") as f:
    return [json.loads(line) for line in f]


def rank0_print(rank, *args, **kwargs):
  if rank == 0:
    print(*args, **kwargs)


def load_local_dataset(
    dataset_path
) -> list[dict]:
  """
  Load dataset from a JSON or JSONL file.
  """
  if dataset_path.endswith('.jsonl'):
    return read_jsonl(dataset_path)
  elif dataset_path.endswith('.json'):
    with open(dataset_path, 'r') as f:
      return json.load(f)
  else:
    return datasets.load_from_disk(dataset_path)


def get_video_frames(
    video_file_path,
    vid_proc_args: ProcessingArguments,
):
  if vid_proc_args is None:
    raise ValueError("Processing Argument must be provided.")
  
  if not os.path.exists(video_file_path):
    print(f"File not exist: {video_file_path}")
    return None
  vr = VideoReader(video_file_path, num_threads=4)
  total_frames = len(vr)
  avg_fps = vr.get_avg_fps()
  
  base_interval = vid_proc_args.base_interval
  min_frames = vid_proc_args.video_min_frames
  max_frames = vid_proc_args.video_max_frames
  
  if not (base_interval and min_frames and max_frames):
    return vr.get_batch(range(total_frames)).asnumpy(), avg_fps
  
  video_length = total_frames / avg_fps
  num_frames_to_sample = round(video_length / base_interval)

  target_frames = min(
      max(num_frames_to_sample, min_frames), max_frames
  )

  frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
  video = vr.get_batch(frame_idx).asnumpy()
  return video, target_frames / video_length


def get_images_and_videos(
    conversation: list[dict],
    vid_proc_args: ProcessingArguments,
) -> tuple[list[str], list[str]]:
  """Conversation should be like 
  [
      {
          "role": "user",
          "content": [
              {"type": "text", "text": "What is this?"},
              {"type": "image", "image": "/absolute/path/to/image1.jpg" | PIL.Image.Image | Base64 string},
              {"type": "video", "video": "/absolute/path/to/video1.mp4"}
          ]
      },
      ...
  ]
  """
  images = list()
  videos = list()
  fpss = list()
  for message in conversation:
    if message['role'] == 'system':
      continue
    for content in message['content']:
      if content['type'] == 'image':
        images.append(get_image(content['image']))
      elif content['type'] == 'video':
        frames, fps = get_video_frames(
          content['video'],
          vid_proc_args,
        )
        videos.append(frames)
        fpss.append(fps)
  return (images, videos, fpss)


def get_image(image) -> Image.Image:
  if isinstance(image, Image.Image):
    return image.convert("RGB")
  if isinstance(image, str) and image.startswith("data:image"):
    image_data = base64.b64decode(image.split(",")[1])
    return Image.open(BytesIO(image_data)).convert("RGB")
  if isinstance(image, str) and os.path.exists(image):
    return Image.open(image).convert("RGB")
  raise ValueError(f"Unsupported image type: {type(image)}. Expected PIL.Image, Base64 string, or file path.")


def make_labels(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    ignore_idx: int = -100,
    assistant_start: str = "<|im_start|>assistant\n",
    assistant_end: str = "<|im_end|>"
) -> torch.Tensor:

  ignore_idx = -100
  assistant_start = "<|im_start|>assistant\n"
  assistant_end = "<|im_end|>"
  labels = torch.ones_like(input_ids) * ignore_idx
  a_start_ids = tokenizer.encode(assistant_start, return_tensors='pt')[0]
  a_end_ids = tokenizer.encode(assistant_end, return_tensors='pt')[0]
  assert len(a_end_ids) == 1
  a_end_idx = list(zip(*torch.where(input_ids == a_end_ids[0])))
  a_start_idx = list(zip(*torch.where(input_ids == a_start_ids[0])))
  assert len(a_start_idx) == len(a_end_idx)
  for (batch_idx, start_idx), (batch_idx, end_idx) in zip(a_start_idx, a_end_idx):
    assert batch_idx == batch_idx, "Batch indices do not match"
    assert start_idx < end_idx, "Start index must be less than end index"
    start_end_idx = start_idx + len(a_start_ids)
    if torch.equal(input_ids[batch_idx, start_idx:start_end_idx], a_start_ids):
      labels[batch_idx, start_end_idx:end_idx] = input_ids[batch_idx, start_end_idx:end_idx]
      
  return labels


def get_batch_images_and_videos(
    conversations,
    vid_proc_args: ProcessingArguments,
) -> tuple[list[str], list[str]]:
  images = list()
  videos = list()
  fpss = list()
  if isinstance(conversations[0], dict):
    conversations = [conversations]
  for conversation in conversations:
    imgs, vids, fps = get_images_and_videos(
        conversation, vid_proc_args
    )
    images.extend(imgs)
    videos.extend(vids)
    fpss.extend(fps)
  return images, videos, fpss


def make_model_input(
    conversations,
    processor,
    proc_args: ProcessingArguments = None,
    add_labels=True,
):
  if add_labels and proc_args.add_generation_prompt:
    raise ValueError("Labels are for finetuning, generation prompt is for inference. Choose one.")
  
  image_inputs, video_inputs, fpss = get_batch_images_and_videos(
      conversations, proc_args
  )
  text = processor.apply_chat_template(
      conversations, add_generation_prompt=proc_args.add_generation_prompt, tokenize=False
  )
  data_dict = processor(
      text=text,
      images=image_inputs if image_inputs else None,
      videos=video_inputs if video_inputs else None,
      fps=fpss if fpss else None,
      return_tensors="pt",
      padding=True,
      padding_side=proc_args.padding_side
  )
  if add_labels:
    data_dict['labels'] = make_labels(
      data_dict['input_ids'],
      processor.tokenizer
    )
  return data_dict


def was_processed_the_same_way(
    dataset: datasets.Dataset,
    saved_path: Path,
    proc_args: ProcessingArguments,
) -> bool:
  if 'num_tokens' not in dataset.features:
    print("Dataset does not have 'num_tokens' feature.")
    return False
  
  try:
    with open(saved_path / 'proc_args.json', 'r') as f:
      og_proc_args = ProcessingArguments(**json.load(f))
    if og_proc_args != proc_args:
      print("Processing arguments have changed.")
      return False
    
  except Exception as e:
    print(f"Error loading metadata: {e}")
    return False
  
  return True


def get_media_names(
    item,
    keys
) -> list[str]:
  media_names = list()
  for key in keys:
    names = item.get(key, list())
    if isinstance(names, str):
      names = [names]
    media_names.extend(names)
  return media_names



def get_num_content_tokens(
    dataset_dir: str,
    media_dir: str,
    processor: AutoProcessor,
    proc_args: ProcessingArguments,
    get_content_fn: Callable,
    num_proc: int = 2,
) -> datasets.Dataset:
  dataset = datasets.load_from_disk(dataset_dir)
  
  if was_processed_the_same_way(dataset, Path(dataset_dir), proc_args):
    print(f"Dataset {dataset_dir} already processed with the same processor and video args.")
    return dataset
  
  def _get_num_content_tokens(item):
    texts, images, videos = get_content_fn(item)
    
    if not (texts or images or videos):
      raise ValueError("Say what now? No content found in item. Check your get_content_fn.")
    
    num_tokens = 0
    for text in texts:
      num_tokens += len(processor.tokenizer.encode(text))
      
    for img in images:
      img = get_image(str(Path(media_dir) / img))
      if img.height < 28 or img.width < 28:
        continue
      result = processor.image_processor(img)
      num_tokens += result['image_grid_thw'].prod() // 4
    
    for vid in videos:
      vid, fps = get_video_frames(str(Path(media_dir) / vid), proc_args)
      if vid.shape[1] < 28 or vid.shape[2] < 28:
        continue
      result = processor.video_processor(vid, fps=fps)
      num_tokens += result['video_grid_thw'].prod() // 4
    
    item['num_tokens'] = num_tokens
    return item

  ds = dataset.map(
    _get_num_content_tokens,
    num_proc=num_proc,
    desc="vision tokens",
    keep_in_memory=True
  )
  ds.save_to_disk(dataset_dir)
  dataset_dir = Path(dataset_dir)
  with open(dataset_dir / 'proc_args.json', 'w') as f:
    json.dump(proc_args.__dict__, f, indent=2)
  return ds


def get_num_tokens(
  ds: datasets.Dataset,
  make_conversation_fn,
  tokenizer,
  num_proc=32
):
  if 'num_tokens' in ds.features:
    return ds['num_tokens']

  def _get_num_tokens(item):
    item['num_tokens'] = item['num_tokens']
    tokens = tokenizer.apply_chat_template(
      make_conversation_fn(item))
    placeholder_count = 0
    for id in tokens:
      if id in PLACEHOLDER_IDS:
        placeholder_count += 1

    item['num_tokens'] += len(tokens) - placeholder_count
    return item

  return ds.map(
      _get_num_tokens,
      num_proc=num_proc,
      desc="total tokens",
  )['num_tokens']
