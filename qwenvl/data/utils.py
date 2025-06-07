import copy
from functools import partial
import json
import os
from pathlib import Path
import random
import shutil
from typing import Callable
import av
import cv2
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
torch.set_num_threads(1)


def _process_video_with_decord(
  video_path: str,
  vid_proc_args: ProcessingArguments
) -> tuple[np.ndarray, float]:
  
  vr = VideoReader(video_path, num_threads=4)
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


def _process_video_with_opencv(
  video_path: str,
  vid_proc_args: ProcessingArguments
) -> tuple[np.ndarray, float]:
  """
  Processes video using OpenCV. Raises an exception on failure.
  """
  cap = cv2.VideoCapture(video_path)
  if not cap.isOpened():
    raise IOError("Cannot open video file with OpenCV")

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  avg_fps = cap.get(cv2.CAP_PROP_FPS)

  if total_frames == 0 or avg_fps == 0:
    cap.release()
    raise ValueError("Video file has zero frames or zero FPS with OpenCV.")

  # If arguments for sampling are not provided, return all frames
  if not all([vid_proc_args.base_interval, vid_proc_args.video_min_frames, vid_proc_args.video_max_frames]):
    frames = []
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
      raise ValueError("Could not extract any frames using OpenCV.")
    return np.stack(frames), avg_fps

  video_length = total_frames / avg_fps
  num_frames_to_sample = round(video_length / vid_proc_args.base_interval)
  target_frames = min(
    max(num_frames_to_sample, vid_proc_args.video_min_frames),
    vid_proc_args.video_max_frames
  )

  frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
  
  frames = []
  for i in frame_idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
      frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  
  cap.release()

  if not frames:
    raise ValueError("Could not extract any frames using OpenCV.")
    
  video = np.stack(frames)
  effective_fps = target_frames / video_length
  return video, effective_fps

def _process_video_with_pyav(
  video_path: str,
  vid_proc_args: ProcessingArguments
) -> tuple[np.ndarray, float]:
  """
  Processes video using PyAV. Raises an exception on failure.
  """
  with av.open(video_path) as container:
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    total_frames = stream.frames
    avg_fps = stream.average_rate

    if total_frames == 0 or avg_fps == 0:
      raise ValueError("Video file has zero frames or zero FPS with PyAV.")

    if not all([vid_proc_args.base_interval, vid_proc_args.video_min_frames, vid_proc_args.video_max_frames]):
      frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
      if not frames:
        raise ValueError("Could not extract any frames using PyAV.")
      return np.stack(frames), avg_fps

    video_length = total_frames / avg_fps
    num_frames_to_sample = round(video_length / vid_proc_args.base_interval)
    target_frames = min(
      max(num_frames_to_sample, vid_proc_args.video_min_frames),
      vid_proc_args.video_max_frames
    )

    frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    
    frames = []
    time_base = stream.time_base
    for i in frame_idx:
      timestamp = int(i * avg_fps.denominator * time_base.numerator / (avg_fps.numerator * time_base.denominator))
      container.seek(timestamp, any_frame=False, backward=True, stream=stream)
      for frame in container.decode(video=0):
        if frame.pts >= timestamp:
          frames.append(frame.to_ndarray(format='rgb24'))
          break
    
    if not frames:
      raise ValueError("Could not extract any frames using PyAV.")
      
    video = np.stack(frames)
    effective_fps = target_frames / video_length
    return video, effective_fps

def get_video_frames(
  video_file_path: str,
  vid_proc_args: ProcessingArguments,
  media_dir: Path = None
) -> tuple[np.ndarray, float]:
  """
  Samples frames from a video file with a robust fallback mechanism.
  
  It first tries using OpenCV for speed and falls back to PyAV for robustness.
  """
  if vid_proc_args is None:
    raise ValueError("ProcessingArguments must be provided.")

  full_path = Path(media_dir) / video_file_path if media_dir else Path(video_file_path)

  if not full_path.exists():
    print(f"File does not exist: {full_path}")
    return None

  video_str_path = str(full_path)

  # try:
  #   return _process_video_with_decord(video_str_path, vid_proc_args)
  # except Exception as e:
  #   print(f"Decord failed: {e}. Falling back to opencv.")
  try:
    return _process_video_with_opencv(video_str_path, vid_proc_args)
  except Exception as e_opencv:
    print(f"OpenCV failed: {e_opencv}. Falling back to PyAV.")
    try:
      return _process_video_with_pyav(video_str_path, vid_proc_args)
    except Exception as e_pyav:
      print(f"PyAV also failed: {e_pyav}")
      return None
    
  
def get_images_and_videos(
    conversation: list[dict],
    vid_proc_args: ProcessingArguments,
    media_dir: Path = None
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
        images.append(get_image(content['image'], media_dir))
      elif content['type'] == 'video':
        frames, fps = get_video_frames(
          content['video'],
          vid_proc_args,
          media_dir
        )
        videos.append(frames)
        fpss.append(fps)
  return (images, videos, fpss)


def get_image(image, media_dir: Path = None) -> Image.Image:
  if isinstance(image, Image.Image):
    return image.convert("RGB")
  if isinstance(image, str) and image.startswith("data:image"):
    image_data = base64.b64decode(image.split(",")[1])
    return Image.open(BytesIO(image_data)).convert("RGB")
  if media_dir is not None:
    image = media_dir / image
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
      labels[batch_idx, start_end_idx:end_idx] = input_ids[batch_idx, start_end_idx:end_idx]
      
  return labels


def get_batch_images_and_videos(
    conversations,
    vid_proc_args: ProcessingArguments,
    media_dir: Path = None
) -> tuple[list[str], list[str]]:
  images = list()
  videos = list()
  fpss = list()
  if not isinstance(conversations, list):
    conversations = [conversations]
  for conversation in conversations:
    imgs, vids, fps = get_images_and_videos(
        conversation, vid_proc_args, media_dir
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
    media_dir: Path = None,
):
  if add_labels and proc_args.add_generation_prompt:
    raise ValueError("Labels are for finetuning, generation prompt is for inference. Choose one.")
  
  image_inputs, video_inputs, fpss = get_batch_images_and_videos(
      conversations, proc_args, media_dir
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


def filter_image(item, image_key='image', id_key=None):
  try:
    image = Image.open(BytesIO(item[image_key]['bytes']))
    image.verify()
    if image.height < 28 or image.width < 28:
      return False
  except Exception as e:
    print(f"Error processing image {item.get(id_key, 'unknown')}: {e}")
    return False

  return True


def processed_the_same(
    saved_path: Path,
    proc_args: ProcessingArguments,
    check_model_length: bool = False
) -> bool:
  """
  Return whether the proc_args.json file in the saved_path matches the provided proc_args.
  If there is an exception while loading the file, return False.
  """
  if proc_args is None:
    return False
  proc_args_d = proc_args.__dict__
  
  try:
    with open(saved_path / 'proc_args.json', 'r') as f:
      og_proc_args_d = ProcessingArguments(**json.load(f)).__dict__
    for k in ['padding_side', 'shortest_edge']:
      proc_args_d.pop(k, None)
      og_proc_args_d.pop(k, None)
      
    if not check_model_length:
      proc_args_d.pop('model_max_length', None)
      og_proc_args_d.pop('model_max_length', None)
      
    if og_proc_args_d != proc_args_d:
      return False
    
  except Exception as e:
    print(f"Error loading metadata: {e}")
    return False
  
  return True


def get_media(
    item,
    keys
) -> list[str]:
  media_names = list()
  for key in keys:
    names = item.get(key, list())
    if not isinstance(names, list):
      names = [names]
    media_names.extend(names)
  return media_names


def get_num_content_tokens(
    dataset: datasets.Dataset,
    media_dir: Path,
    processor: AutoProcessor,
    proc_args: ProcessingArguments,
    get_content_fn: Callable,
    num_proc: int = 2,
) -> datasets.Dataset:
  def _get_num_content_tokens(item: dict) -> dict:
    texts, images, videos = get_content_fn(item)

    num_tokens = 0
    for text in texts:
      num_tokens += len(processor.tokenizer.encode(text))
  
    for img in images:
      img = get_image(img, media_dir)
      result = processor.image_processor(img)
      num_tokens += result['image_grid_thw'].prod() // 4
    
    for vid in videos:
      vid, fps = get_video_frames(vid, proc_args, media_dir)
      result = processor.video_processor(vid, fps=fps)
      num_tokens += result['video_grid_thw'].prod() // 4
    
    item['num_content_tokens'] = num_tokens
    return item

  mapped = dataset.map(
    _get_num_content_tokens,
    num_proc=num_proc,
    desc="content tokens",
  )
  return mapped

def save_w_proc_args(
    dataset: datasets.Dataset,
    dataset_dir: Path,
    proc_args: ProcessingArguments,
) -> None:
  dataset.save_to_disk(dataset_dir)
  with open((dataset_dir) / 'proc_args.json', 'w') as f:
    json.dump(proc_args.__dict__, f, indent=2)


def get_num_tokens(
  ds: datasets.Dataset,
  make_conversation_fn,
  tokenizer,
  num_proc=32
):
  if 'num_tokens' in ds.features:
    return ds['num_tokens']

  def _get_num_tokens(item):
    tokens = tokenizer.apply_chat_template(
      make_conversation_fn(item))
    placeholder_count = 0
    for id in tokens:
      if id in PLACEHOLDER_IDS:
        placeholder_count += 1

    item['num_tokens'] = len(tokens) - placeholder_count + item['num_content_tokens']
    return item

  return ds.map(
      _get_num_tokens,
      num_proc=num_proc,
      desc="total tokens",
  )['num_tokens']
