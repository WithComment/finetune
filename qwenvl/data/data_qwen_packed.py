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

from qwenvl.common_utils import make_model_input, get_images_and_videos, load_datasets, make_labels, process_image, process_video, rank0_print

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None


def read_jsonl(path):
  with open(path, "r") as f:
    return [json.loads(line) for line in f]


class LazySupervisedDataset(Dataset):
  """Dataset for supervised fine-tuning."""

  def __init__(self, processor: transformers.AutoProcessor, data_args):
    super(LazySupervisedDataset, self).__init__()
    
    list_data_dict = load_datasets(data_args.dataset_use.split(","), local_rank)
    
    rank0_print(local_rank, f"Total training samples: {len(list_data_dict)}")

    random.shuffle(list_data_dict)  # Randomly shuffle the data for training

    self.processor = copy.deepcopy(processor)
    self.list_data_dict = list_data_dict
    for attr, val in vars(data_args).items():
      setattr(self, attr, val)

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
    

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    return self.list_data_dict[i]

  def get_data(self, i) -> Dict[str, torch.Tensor]:
    conversation = [{
        "role": "system",
        "content": "You are a student in medicine studying for a final exam."
    }] + self.list_data_dict[i]
    return make_model_input(
      conversation, self.processor, self.base_interval, self.video_min_frames, self.video_max_frames)


@dataclass
class PackedDataCollatorForSupervisedDataset(object):
  """Collate examples into packed sequence with multi-modal support."""
  processor: transformers.PreTrainedTokenizer
  base_interval: int = 2
  video_min_frames: int = 4
  video_max_frames: int = 8

  def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    sys_msg = [{
        "role": "system",
        "content": "You are a student in medicine studying for a final exam."
    }]
    instances = [make_model_input(
      sys_msg + instance, self.processor, self.base_interval,
      self.video_min_frames, self.video_max_frames, add_generation_prompt=False
    ) for instance in instances]
    input_ids, labels, attention_mask = tuple(
        [instance[key] for instance in instances]
        for key in ("input_ids", "labels", "attention_mask")
    )
    attention_mask = list(
        itertools.chain(
            *(
                instance["attention_mask"][0]
                for instance in instances
                if "attention_mask" in instance
            )
        )
    )
    seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
    cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
    input_ids = torch.cat(input_ids, dim=1)
    labels = torch.cat(labels, dim=1)

    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=cumsum_seq_lens,
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
      processor=processor)
  return dict(
      train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
  )


if __name__ == "__main__":
  pass
