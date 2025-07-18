import json
import math
from pathlib import Path
import subprocess
from typing import Callable
import cv2
import datasets
from decord import VideoReader
import numpy as np
from PIL import Image
import webvtt

import torch

from transformers import AutoProcessor, PreTrainedTokenizer, Qwen2_5_VLProcessor

import base64
from io import BytesIO

from ..argument import ProcessingArguments

from ..utils import get_logger

logger = get_logger(__name__)

PLACEHOLDER_IDS = {151654, 151655, 151656}
torch.set_num_threads(1)


def make_cot(subtitle_file: Path):
  if not subtitle_file.exists():
    return ''
  cot = "### Thinking frame by frame:\n\n"
  sub = webvtt.read(subtitle_file)
  for caption in sub:
    cot += f"<timestamp>: {caption.start} --> {caption.end}\n"
    cot += f"<content>: {caption.text}\n\n"
  cot += "### End thinking\n\n"
  return cot


def filter_cot(item: dict, subtitle_dir: Path) -> bool:
  cot = subtitle_dir / item['video'].replace('.mp4', '.en.vtt')
  return cot.exists()


def get_vid_frames_opencv(
    video_path: str,
    vid_proc_args: ProcessingArguments,
    all_frames: bool,
    is_checking: bool,
    is_counting: bool,
) -> tuple[np.ndarray, float]:
  """
  Processes video using OpenCV. Raises an exception on failure.
  """
  cap = cv2.VideoCapture(video_path)
  
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  avg_fps = cap.get(cv2.CAP_PROP_FPS)

  if total_frames == 0 or avg_fps == 0:
    cap.release()
    raise ValueError(f"Video file {video_path} has zero frames or zero FPS with OpenCV.")
  if is_checking:
    return
  
  video_length = total_frames / avg_fps
  if all_frames:
    target_frames = total_frames
  else:
    num_frames_to_sample = round(video_length / vid_proc_args.base_interval)
    target_frames = min(
        max(num_frames_to_sample, vid_proc_args.video_min_frames),
        vid_proc_args.video_max_frames
    )
    
  frames = []
  if is_counting:
    ret, frame = cap.read()
    if ret:
      frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  
  else:
    frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)

    for i in frame_idx:
      cap.set(cv2.CAP_PROP_POS_FRAMES, i)
      ret, frame = cap.read()
      if ret:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

  cap.release()

  if not frames:
    raise ValueError("Could not extract any frames using OpenCV.")

  video = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
  effective_fps = target_frames / video_length
  return video, effective_fps


def get_video_frames(
    video: Path,
    vid_proc_args: ProcessingArguments,
    is_checking: bool = False,
    is_counting: bool = False,
) -> tuple[torch.Tensor, float]:
  """
  Samples frames from a video file with a robust fallback mechanism.

  It first tries using OpenCV for speed and falls back to PyAV for robustness.
  """
  if isinstance(video, str):
    video = Path(video)
  all_frames = video.parent.name == 'vid_processed'
  return get_vid_frames_opencv(video, vid_proc_args, all_frames, is_checking, is_counting)


def get_image(image) -> Image.Image:
  if isinstance(image, Image.Image):
    return image.convert("RGB")
  if isinstance(image, str) and image.startswith("data:image"):
    image_data = base64.b64decode(image.split(",")[1])
    return Image.open(BytesIO(image_data)).convert("RGB")
  return Image.open(image).convert("RGB")


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
      if "image" in content:
        images.append(get_image(content['image']))
      elif "video" in content:
        frames, fps = get_video_frames(content['video'], vid_proc_args)
        videos.append(frames)
        fpss.append(fps)
  return (images, videos, fpss)



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


def verify_video(item, media_dir):
  video_path = media_dir / item['video']
  return video_path.exists()


def reencode(item, media_dir, logger):
  output_path = media_dir.parent / 'vid_processed' / item['video']
  video_path = media_dir / item['video']
  if output_path.exists():
    item['status'] = 'Already Processed'
    logger.info(f"✅ Already processed: {video_path}")
    return item
  if not video_path.exists():
    item['status'] = 'DNE'
    return item

  cmd = [
      "ffmpeg",
      "-y",
      "-i", str(video_path),
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", "23",
      "-c:a", "copy",
      str(output_path)
  ]
  try:
    result = subprocess.run(cmd, stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    if result.returncode == 0:
      logger.info(f"✅ Successfully re-encoded: {video_path}")
    else:
      logger.warning(f"❌ Failed to re-encode: {video_path}")
      logger.debug(result.stderr)
      item['status'] = 'Corrupted'
      return item

  except Exception as e:
    logger.error(f"🚨 Error processing {video_path}: {e}")
    item['status'] = 'PyERROR'
    return item

  item['status'] = 'OK'
  return item


def round_by_factor(number: int, factor: int) -> int:
  """Returns the closest integer to 'number' that is divisible by 'factor'."""
  return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
  """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
  return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
  """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
  return math.floor(number / factor) * factor


def smart_resize(
    height: int, width: int, max_pixels: int, min_pixels: int
) -> tuple[int]:
  """
  Rescales the image so that the following conditions are met:

  1. Both dimensions (height and width) are divisible by 'factor'.

  2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

  3. The aspect ratio of the image is maintained as closely as possible.
  """
  factor = 28
  h_bar = max(factor, round_by_factor(height, factor))
  w_bar = max(factor, round_by_factor(width, factor))
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = max(factor, floor_by_factor(height / beta, factor))
    w_bar = max(factor, floor_by_factor(width / beta, factor))
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = ceil_by_factor(height * beta, factor)
    w_bar = ceil_by_factor(width * beta, factor)
  return h_bar, w_bar, h_bar // factor, w_bar // factor
