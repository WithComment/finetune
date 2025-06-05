from functools import partial
import copy
import json
import random
from typing import Dict, Sequence
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers

from qwenvl.data import data_list
from qwenvl.data.utils import load_local_dataset, make_model_input, rank0_print
from qwenvl.data.openpmc import OpenpmcDataset
from qwenvl.train.argument import DataArguments, ProcessingArguments


local_rank = None

dataset_classes = {
    'openpmc': OpenpmcDataset
}

def read_jsonl(path):
  with open(path, "r") as f:
    return [json.loads(line) for line in f]


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


class LazySupervisedDataset(Dataset):
  """Dataset for supervised fine-tuning."""

  def __init__(
      self,
      processor: transformers.AutoProcessor,
      data_args: DataArguments,
      proc_args: ProcessingArguments,
  ):
    super(LazySupervisedDataset, self).__init__()

    list_data_dict = load_datasets(
        data_args.dataset_use.split(","), local_rank)

    rank0_print(
        local_rank, f"Total training samples: {len(list_data_dict)}")

    random.shuffle(list_data_dict)  # Randomly shuffle the data for training

    self.processor = copy.deepcopy(processor)
    self.list_data_dict = list_data_dict
    self.proc_args = proc_args
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
    if "token_count" in self.list_data_dict[0]:
      length_list = [sample["token_count"] for sample in self.list_data_dict]
      return np.array(length_list)
    else:
      print("No pre-calculated length available.")
      return np.array([1] * len(self.list_data_dict))

  def __getitem__(self, i) -> Dict[str, torch.Tensor]:
    return self.list_data_dict[i]['conversation']


def make_batch_input(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
  return make_model_input(
      instances,
      self.processor,
      self.proc_args,
  )


def make_supervised_data_module(
    processor: transformers.AutoProcessor,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
) -> Dict:
  """Make dataset and collator for supervised fine-tuning."""
  dataset_config = data_list(data_args.dataset_use.split(","))[0]
  train_dataset = dataset_classes[dataset_config['dataset_class']](
      media_dir=data_args.media_dir,
      processor=processor,
      proc_args=proc_args,
      use_cft=data_args.use_cft,
      dataset_dir=data_args.dataset_dir,
      pack=data_args.data_packing,
  )
  collate_fn = partial(
      make_model_input,
      processor=processor,
      proc_args=proc_args,
      add_labels=True
  )
  return dict(
      train_dataset=train_dataset, eval_dataset=None, data_collator=collate_fn
  )
