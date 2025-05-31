import copy
from functools import partial
import json
import os
import random
import datasets
from decord import VideoReader
import numpy as np
from PIL import Image

import torch

from qwenvl.data import data_list

from transformers import PreTrainedTokenizer

import base64
from io import BytesIO
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


def load_datasets(
    dataset_names: list[str],
    rank=0,
) -> list[dict]:
  list_data_dict = []

  dataset_list = data_list(dataset_names)
  rank0_print(rank, f"Loading datasets: {dataset_list}")
  for data in dataset_list:
    annotations = load_local_dataset(data["dataset_path"])
    sampling_rate = data['sampling_rate']
    annotations = random.sample(
        annotations, int(len(annotations) * sampling_rate)
    )
    rank0_print(
        rank, f"sampling {len(annotations)} examples from dataset {data}")
    list_data_dict += annotations
  return list_data_dict


def get_video_frames(
    video_file_path,
    base_interval,
    min_frames,
    max_frames,
):
  if not os.path.exists(video_file_path):
    print(f"File not exist: {video_file_path}")
    return None
  vr = VideoReader(video_file_path, num_threads=4)
  total_frames = len(vr)
  avg_fps = vr.get_avg_fps()
  
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


def process_video_frames(processor, video, fps):
  video_processed = processor.preprocess(
      videos=video, return_tensors="pt"
  )
  video_tensor = video_processed["pixel_values_videos"]
  grid_thw = video_processed["video_grid_thw"][0]
  second_per_grid_ts = [
      processor.temporal_patch_size / fps
  ] * len(grid_thw)
  return video_tensor, grid_thw, second_per_grid_ts


def process_video(
    video_file_path,
    processor,
    base_interval,
    min_frames,
    max_frames,
):
  video_frames, fps = get_video_frames(
      video_file_path, base_interval, min_frames, max_frames
  )

  return process_video_frames(processor, video_frames, fps)


def process_image(
    image_file_path,
    processor
) -> tuple:
  image = Image.open(image_file_path).convert("RGB")
  visual_processed = processor.preprocess(image, return_tensors="pt")
  image_tensor = visual_processed["pixel_values"]
  if isinstance(image_tensor, list):
    image_tensor = image_tensor[0]
  grid_thw = visual_processed["image_grid_thw"][0]
  return image_tensor, grid_thw


def get_images_and_videos(
    conversation: list[dict],
    base_interval: int,
    min_frames: int,
    max_frames: int,
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
          base_interval=base_interval,
          min_frames=min_frames,
          max_frames=max_frames
        )
        videos.append(frames)
        fpss.append(fps)
  return (images, videos, fpss)


def get_image(image):
  if isinstance(image, Image.Image):
    return image.convert("RGB")
  if isinstance(image, str) and image.startswith("data:image"):
    image_data = base64.b64decode(image.split(",")[1])
    return Image.open(BytesIO(image_data)).convert("RGB")
  if isinstance(image, str) and os.path.exists(image):
    return Image.open(image).convert("RGB")


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
      end_idx += 1 # Adjust end index to include the end token
      labels[batch_idx, start_end_idx + 1:end_idx] = input_ids[batch_idx, start_end_idx + 1:end_idx]
      
  return labels


def get_batch_images_and_videos(
    conversations,
    base_interval,
    min_frames,
    max_frames,
) -> tuple[list[str], list[str]]:
  images = list()
  videos = list()
  fpss = list()
  if isinstance(conversations[0], dict):
    conversations = [conversations]
  for conversation in conversations:
    imgs, vids, fps = get_images_and_videos(
        conversation, base_interval, min_frames, max_frames
    )
    images.extend(imgs)
    videos.extend(vids)
    fpss.extend(fps)
  return images, videos, fpss


def make_model_input(
    conversations,
    processor,
    base_interval,
    video_min_frames,
    video_max_frames,
    add_generation_prompt=True,
    padding_side='right',
):
    
  image_inputs, video_inputs, fpss = get_batch_images_and_videos(
      conversations, base_interval, video_min_frames, video_max_frames
  )
  text = processor.apply_chat_template(
      conversations, add_generation_prompt=add_generation_prompt, tokenize=False
  )
  data_dict = processor(
      text=text,
      images=image_inputs if image_inputs else None,
      videos=video_inputs if video_inputs else None,
      fps=fpss if fpss else None,
      return_tensors="pt",
      padding=True,
      padding_side=padding_side
  )
  data_dict['labels'] = make_labels(
    data_dict['input_ids'],
    processor.tokenizer
  )
  return data_dict