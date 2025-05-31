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
    return self.list_data_dict[i]['conversation']
    return self.get_data(i)

  def get_data(self, i) -> Dict[str, torch.Tensor]:
    return make_model_input(
      self.list_data_dict[i]['conversation'],
      self.processor,
      self.base_interval,
      self.video_min_frames,
      self.video_max_frames,
      add_generation_prompt=False
    )


@dataclass
class PackedDataCollatorForSupervisedDataset(object):
  """Collate examples into packed sequence with multi-modal support."""
  processor: transformers.PreTrainedTokenizer
  base_interval: int = 2
  video_min_frames: int = 4
  video_max_frames: int = 8
  
  def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    return make_model_input(
      instances,
      self.processor,
      self.base_interval,
      self.video_min_frames,
      self.video_max_frames,
      add_generation_prompt=False,
    )


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
