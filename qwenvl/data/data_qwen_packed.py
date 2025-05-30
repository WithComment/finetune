from functools import partial
import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from decord import VideoReader
import transformers

from qwenvl.common_utils import process_video

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def rank0_print(*args):
  if local_rank == 0:
    print(*args)


def read_jsonl(path):
  with open(path, "r") as f:
    return [json.loads(line) for line in f]


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
  roles = {"human": "user", "gpt": "assistant"}
  system_message = "You are a helpful assistant."

  tokenizer = copy.deepcopy(tokenizer)
  chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  tokenizer.chat_template = chat_template

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


class LazySupervisedDataset(Dataset):
  """Dataset for supervised fine-tuning."""

  def __init__(self, processor: transformers.AutoProcessor, data_args):
    super(LazySupervisedDataset, self).__init__()

    dataset = data_args.dataset_use.split(",")
    dataset_list = data_list(dataset)
    rank0_print(f"Loading datasets: {dataset_list}")
    self.model_type = data_args.model_type
    if data_args.model_type == "qwen2.5vl":
      self.get_rope_index = get_rope_index_25
    else:
      self.get_rope_index = get_rope_index_2

    list_data_dict = []

    for data in dataset_list:
      file_format = data["dataset_path"].split(".")[-1]
      if file_format == "jsonl":
        annotations = read_jsonl(data["dataset_path"])
      else:
        annotations = json.load(open(data["dataset_path"], "r"))
      sampling_rate = data.get("sampling_rate", 1.0)
      if sampling_rate < 1.0:
        annotations = random.sample(
            annotations, int(len(annotations) * sampling_rate)
        )
        print(f"sampling {len(annotations)} examples from dataset {data}")
      else:
        rank0_print(f"dataset name: {data}")
      for ann in annotations:
        if isinstance(ann, list):
          for sub_ann in ann:
            sub_ann["media_dir"] = data["media_dir"]
        else:
          ann["media_dir"] = data["media_dir"]
      list_data_dict += annotations

    rank0_print(f"Total training samples: {len(list_data_dict)}")

    random.shuffle(list_data_dict)  # Randomly shuffle the data for training

    rank0_print("Formatting inputs...Skip in lazy mode")
    self.tokenizer = copy.deepcopy(processor.tokenizer)
    self.image_processor = copy.deepcopy(processor.image_processor)
    self.video_processor = copy.deepcopy(processor.video_processor)
    self.list_data_dict = list_data_dict
    for attr, val in vars(data_args).items():
      setattr(self, attr, val)
    self.process_video = partial(
        process_video,
        processor=self.video_processor,
        base_interval=self.base_interval, 
        min_frames=self.video_min_frames, 
        max_frames=self.video_max_frames,
    )

  def __len__(self):
    return len(self.list_data_dict)

  @property
  def lengths(self):
    length_list = []
    for sample in self.list_data_dict:
      img_tokens = 128 if "image" in sample else 0
      length_list.append(
          sum(len(conv["value"].split()) for conv in sample["conversations"])
          + img_tokens
      )
    return length_list

  @property
  def modality_lengths(self):
    length_list = []
    for sample in self.list_data_dict:
      cur_len = sum(
          len(conv["value"].split()) for conv in sample["conversations"]
      )
      cur_len = (
          cur_len if ("image" in sample) or ("video" in sample) else -cur_len
      )
      length_list.append(cur_len)
    return length_list

  @property
  def pre_calculated_length(self):
    if "num_tokens" in self.list_data_dict[0]:
      length_list = [sample["num_tokens"] for sample in self.list_data_dict]
      return np.array(length_list)
    else:
      print("No pre-calculated length available.")
      return np.array([1] * len(self.list_data_dict))

  def process_image_unified(self, image_file):
    processor = copy.deepcopy(self.image_processor)
    image = Image.open(image_file).convert("RGB")

    visual_processed = processor.preprocess(image, return_tensors="pt")
    image_tensor = visual_processed["pixel_values"]
    if isinstance(image_tensor, List):
      image_tensor = image_tensor[0]
    grid_thw = visual_processed["image_grid_thw"][0]
    return image_tensor, grid_thw

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    num_base_retries = 3
    num_final_retries = 30

    # try the current sample first
    for attempt_idx in range(num_base_retries):
      try:
        sample = self._get_item(i)
        return sample
      except Exception as e:
        # sleep 1s in case it is a cloud disk issue
        print(
            f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
        time.sleep(1)

    # try other samples, in case it is file corruption issue
    for attempt_idx in range(num_base_retries):
      try:
        next_index = min(i + 1, len(self.list_data_dict) - 1)
        # sample_idx = random.choice(range(len(self)))
        sample = self._get_item(next_index)
        return sample
      except Exception as e:
        # no need to sleep
        print(
            f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
            e,
        )
        pass

    try:
      sample = self._get_item(i)
      return sample
    except Exception as e:
      raise e

  def get_data(self, sources) -> Dict[str, torch.Tensor]:
    # define some variables
    grid_thw_merged = None
    video_grid_thw_merged = None
    grid_thw = None
    video_grid_thw = None
    second_per_grid_ts = None

    if "image" in sources[0]:
      image_folder = sources[0]["media_dir"]
      image_file = sources[0]["image"]
      if isinstance(image_file, List):
        if len(image_file) > 1:
          image_file = [
              os.path.join(image_folder, file) for file in image_file
          ]
          results = [self.process_image_unified(file) for file in image_file]
          image, grid_thw = zip(*results)
        else:
          image_file = image_file[0]
          image_file = os.path.join(image_folder, image_file)
          image, grid_thw = self.process_image_unified(image_file)
          image = [image]
      else:
        image_file = os.path.join(image_folder, image_file)
        image, grid_thw = self.process_image_unified(image_file)
        image = [image]
      grid_thw_merged = copy.deepcopy(grid_thw)
      if not isinstance(grid_thw, Sequence):
        grid_thw_merged = [grid_thw_merged]
        grid_thw = [grid_thw]
      grid_thw_merged = [
          merged_thw.prod() // self.image_processor.merge_size**2
          for merged_thw in grid_thw_merged
      ]
    if "video" in sources[0]:
      video_file = sources[0]["video"]
      video_folder = sources[0]["media_dir"]
      if isinstance(video_file, List):
        if len(video_file) > 1:
          video_file = [
              os.path.join(video_folder, file) for file in video_file
          ]
          results = [self.process_video(file) for file in video_file]
          video, video_grid_thw, second_per_grid_ts = zip(*results)
        else:
          video_file = video_file[0]
          video_file = os.path.join(video_folder, video_file)
          video, video_grid_thw, second_per_grid_ts = self.process_video(
              video_file
          )
          video = [video]
      else:
        video_file = os.path.join(video_folder, video_file)
        video, video_grid_thw, second_per_grid_ts = self.process_video(
            video_file
        )
        video = [video]
      video_grid_thw_merged = copy.deepcopy(video_grid_thw)
      if not isinstance(video_grid_thw, Sequence):
        video_grid_thw_merged = [video_grid_thw_merged]
        video_grid_thw = [video_grid_thw]
      video_grid_thw_merged = [
          merged_thw.prod() // self.image_processor.merge_size**2
          for merged_thw in video_grid_thw_merged
      ]
    chat_sources = copy.deepcopy([e["conversations"] for e in sources])
    data_dict = preprocess_qwen_2_visual(
        chat_sources,
        self.tokenizer,
        grid_thw_image=grid_thw_merged if grid_thw_merged else None,
        grid_thw_video=video_grid_thw_merged if video_grid_thw_merged else None,
    )
    position_ids, _ = self.get_rope_index(
        self.image_processor.merge_size,
        data_dict["input_ids"],
        image_grid_thw=torch.stack(grid_thw, dim=0) if grid_thw else None,
        video_grid_thw=(
            torch.stack(video_grid_thw, dim=0) if video_grid_thw else None
        ),
        second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
    )
    if "image" not in sources[0] and "video" not in sources[0]:
      grid_thw_merged = None
      sources = copy.deepcopy([e["conversations"] for e in sources])
      data_dict = preprocess_qwen_2_visual(
          sources, self.tokenizer, grid_thw=grid_thw_merged
      )
      position_ids = (
          torch.arange(0, data_dict["input_ids"].size(1))
          .view(1, -1)
          .unsqueeze(0)
          .expand(3, -1, -1)
      )

    data_dict["position_ids"] = position_ids
    data_dict["attention_mask"] = [data_dict["input_ids"][0].size(0)]

    if "image" in sources[0]:
      data_dict["pixel_values"] = torch.cat(image, dim=0)
      data_dict["image_grid_thw"] = torch.cat(
          [thw.unsqueeze(0) for thw in grid_thw], dim=0
      )
    # video exist in the data
    elif "video" in sources[0]:
      data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
      data_dict["video_grid_thw"] = torch.cat(
          [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
      )

    return data_dict

  def _get_item(self, i) -> Dict[str, torch.Tensor]:

    sources = self.list_data_dict[i]

    if isinstance(sources, dict):
      if isinstance(i, int):
        sources = [sources]
      assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
      return self.get_data(sources)

    if isinstance(sources, list):
      data_list = []
      new_data_dict = {}
      for source in sources:
        if isinstance(i, int):
          source = [source]
        assert (
            len(source) == 1
        ), "Don't know why it is wrapped to a list"  # FIXME
        data_list.append(self.get_data(source))

      input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
      labels = torch.cat([d["labels"] for d in data_list], dim=1)
      position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
      attention_mask = [
          d["attention_mask"][0] for d in data_list if "attention_mask" in d
      ]
      new_data_dict = {
          "input_ids": input_ids,
          "labels": labels,
          "position_ids": position_ids,
          "attention_mask": attention_mask if attention_mask else None
      }

      if any("pixel_values" in d for d in data_list):
        new_data_dict.update({
            "pixel_values": torch.cat([d["pixel_values"] for d in data_list if "pixel_values" in d], dim=0),
            "image_grid_thw": torch.cat([d["image_grid_thw"] for d in data_list if "image_grid_thw" in d], dim=0)
        })

      if any("pixel_values_videos" in d for d in data_list):
        new_data_dict.update({
            "pixel_values_videos": torch.cat([d["pixel_values_videos"] for d in data_list if "pixel_values_videos" in d], dim=0),
            "video_grid_thw": torch.cat([d["video_grid_thw"] for d in data_list if "video_grid_thw" in d], dim=0)
        })
      return new_data_dict


def pad_and_cat(tensor_list):
  max_length = max(tensor.shape[2] for tensor in tensor_list)

  padded_tensors = []
  for tensor in tensor_list:
    pad_length = max_length - tensor.shape[2]
    padded_tensor = torch.nn.functional.pad(
        tensor, (0, pad_length), "constant", 1)
    padded_tensors.append(padded_tensor)

  stacked_tensor = torch.cat(padded_tensors, dim=1)

  return stacked_tensor


@dataclass
class PackedDataCollatorForSupervisedDataset(object):
  """Collate examples into packed sequence with multi-modal support."""

  tokenizer: transformers.PreTrainedTokenizer

  def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels, position_ids, attention_mask = tuple(
        [instance[key] for instance in instances]
        for key in ("input_ids", "labels", "position_ids", "attention_mask")
    )
    attention_mask = list(
        itertools.chain(
            *(
                instance["attention_mask"]
                for instance in instances
                if "attention_mask" in instance
            )
        )
    )
    seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
    cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    input_ids = torch.cat(input_ids, dim=1)
    labels = torch.cat(labels, dim=1)
    position_ids = torch.cat(position_ids, dim=2)

    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=cumsum_seq_lens,
        position_ids=position_ids,
    )
    images = list(
        instance["pixel_values"]
        for instance in instances
        if "pixel_values" in instance
    )
    videos = list(
        instance["pixel_values_videos"]
        for instance in instances
        if "pixel_values_videos" in instance
    )
    if len(images) != 0:
      concat_images = torch.cat([image for image in images], dim=0)
      grid_thw = [
          instance["image_grid_thw"]
          for instance in instances
          if "image_grid_thw" in instance
      ]
      grid_thw = torch.cat(grid_thw, dim=0)
    else:
      concat_images = None
      grid_thw = None

    if len(videos) != 0:
      concat_videos = torch.cat([video for video in videos], dim=0)
      video_grid_thw = [
          instance["video_grid_thw"]
          for instance in instances
          if "video_grid_thw" in instance
      ]
      video_grid_thw = torch.cat(video_grid_thw, dim=0)
    else:
      concat_videos = None
      video_grid_thw = None

    batch["pixel_values"] = concat_images
    batch["image_grid_thw"] = grid_thw
    batch["pixel_values_videos"] = concat_videos
    batch["video_grid_thw"] = video_grid_thw

    return batch


def make_supervised_data_module_packed(
    processor: transformers.AutoProcessor, data_args
) -> Dict:
  """Make dataset and collator for supervised fine-tuning."""
  train_dataset = LazySupervisedDataset(
      processor=processor, data_args=data_args)
  data_collator = PackedDataCollatorForSupervisedDataset(
      tokenizer=processor.tokenizer)
  return dict(
      train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
  )


if __name__ == "__main__":
  pass
