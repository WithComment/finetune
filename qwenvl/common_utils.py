import copy
import json
import os
import datasets
from decord import VideoReader
import numpy as np

from qwenvl.data import data_list

from transformers import PreTrainedTokenizer


def read_jsonl(path):
  with open(path, "r") as f:
    return [json.loads(line) for line in f]


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


def preprocess_qwen_2_visual(
    sources,
    tokenizer: PreTrainedTokenizer,
    grid_thw_image: list = [],
    grid_thw_video: list = [],
) -> dict:
  roles = {"human": "user", "gpt": "assistant"}
  system_message = "You are a helpful assistant."

  tokenizer = copy.deepcopy(tokenizer)
  
  visual_replicate_index_image = 0
  visual_replicate_index_video = 0
  input_ids, targets = [], []

  for i, source in enumerate(sources):
    try:
      if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]
    except:
      print(sources)

    input_id, target = [], []

    input_id += tokenizer.apply_chat_template(
        [{"role": "system", "content": system_message}]
    )
    target += [IGNORE_INDEX] * len(input_id)

    for conv in source:
      try:
        role = conv["role"]
        content = conv["content"]
      except:
        role = conv["from"]
        content = conv["value"]

      role = roles.get(role, role)
      if role == "user":
        if "<image>" in content:
          parts = content.split("<image>")
          new_parts = []
          for i in range(len(parts) - 1):
            new_parts.append(parts[i])
            replacement = (
                "<|vision_start|>"
                + f"<|image_pad|>"
                * grid_thw_image[visual_replicate_index_image]
                + "<|vision_end|>"
            )
            new_parts.append(replacement)
            visual_replicate_index_image += 1
          new_parts.append(parts[-1])
          content = "".join(new_parts)

        if "<video>" in content:
          parts = content.split("<video>")
          new_parts = []
          for i in range(len(parts) - 1):
            new_parts.append(parts[i])
            replacement = (
                "<|vision_start|>"
                + f"<|video_pad|>"
                * grid_thw_video[visual_replicate_index_video]
                + "<|vision_end|>"
            )
            new_parts.append(replacement)
            visual_replicate_index_video += 1
          new_parts.append(parts[-1])
          content = "".join(new_parts)

      conv = [{"role": role, "content": content}]
      encode_id = tokenizer.apply_chat_template(conv)
      input_id += encode_id
      if role in ["user", "system"]:
        target += [IGNORE_INDEX] * len(encode_id)
      else:
        target_mask = encode_id.copy()
        target_mask[:3] = [IGNORE_INDEX] * 3
        target += target_mask

    assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
    input_ids.append(input_id)
    targets.append(target)

  input_ids = torch.tensor(input_ids, dtype=torch.long)
  targets = torch.tensor(targets, dtype=torch.long)

  return dict(
      input_ids=input_ids,
      labels=targets,
  )
  

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
  video_length = total_frames / avg_fps
  num_frames_to_sample = round(video_length / base_interval)

  target_frames = min(
      max(num_frames_to_sample, min_frames), max_frames
  )
  
  frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
  video = vr.get_batch(frame_idx).asnumpy()
  return video, target_frames / video_length


def process_video_frames(processor, video, fps):
  processor = copy.deepcopy(processor)
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