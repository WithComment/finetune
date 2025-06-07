
from abc import ABC, abstractmethod
import copy
from functools import partial
import json
import os
from pathlib import Path
import random
import datasets
import torch
from torch.utils.data import Dataset
from tqdm import trange
from transformers import AutoProcessor

from qwenvl.data.generic_dataset import GenericDataset
from qwenvl.data.packing import fast_best_fit_decreasing
from qwenvl.data.utils import get_batch_images_and_videos, get_image, get_num_content_tokens, get_num_tokens, get_video_frames, make_model_input, save_w_proc_args, processed_the_same
from qwenvl.train.argument import DataArguments, ProcessingArguments

DEBUG = os.environ.get('DEBUG', '0') == '1'

def sample_from_datadict(
    ds: datasets.DatasetDict,
    sampling_rate: int | float
) -> datasets.DatasetDict:
  sample = datasets.DatasetDict()
  for split_name, split_ds in ds.items():
    if isinstance(sampling_rate, float):
      num = int(len(split_ds) * sampling_rate)
    sample[split_name] = split_ds.select(random.sample(
      range(len(split_ds)), num))
  
  return sample


class SFTDataset(GenericDataset, ABC):
  """
  OpenPMC dataset for training and evaluation.
  """
  generation_config: dict = None

  def __init__(
      self,
      processor: AutoProcessor,
      data_args: DataArguments,
      proc_args: ProcessingArguments,
      sampling_rate: float = 1.0,
  ):
    super(Dataset, self).__init__()
    self.use_cft = data_args.use_cft
    dataset_dir, media_dir = self.get_dataset_dir(self.ds_key)
    
    if sampling_rate < 1.0:
      dataset_dir = dataset_dir.with_name(
        f"{dataset_dir.name}_{sampling_rate}")

    self.dataset_dir = dataset_dir
    self.media_dir = media_dir
    self.processor = copy.deepcopy(processor)
    self.proc_args = copy.deepcopy(proc_args)

    if data_args.count_tokens and not processed_the_same(dataset_dir, proc_args):
      print("(Re)counting tokens.")
      print(f"Downloading dataset {data_args.dataset_use}...")
      
      self.ds = sample_from_datadict(
        self.download(force=False, save_to_disk=False),
        sampling_rate=sampling_rate
      )
      
      self.ds = get_num_content_tokens(
          self.ds,
          media_dir,
          processor=self.processor,
          proc_args=proc_args,
          get_content_fn=self.get_content,
          num_proc=32,
      )
      save_w_proc_args(self.ds, dataset_dir, proc_args)
    else:
      print(f"Loading dataset {dataset_dir} with the same processing arguments.")
      
    self.ds = datasets.load_from_disk(dataset_dir)[data_args.split]
    self.tokenizer = self.processor.tokenizer
    self.make_conversation = partial(
      self._make_conversation,
      media_dir=self.media_dir,
      use_cft=self.use_cft
    )
    if not data_args.data_packing:
      self.bins = None
    else:
      self.bins = self.get_packing_bins(get_num_tokens(
          self.ds,
          self.make_conversation,
          tokenizer=self.tokenizer,
        ), proc_args.model_max_length)
  

  def get_packing_bins(
      self,
      num_tokens: list[int],
      bin_capacity: int,
  ) -> list[list[int]]:
    """
    Get packing bins for dataset.
    Save """
    packing_bins_path = self.dataset_dir / 'packing_bins.json'
    if (
        packing_bins_path.exists() and processed_the_same(
        self.dataset_dir, self.proc_args, check_model_length=True)
    ):
      print(f"Loading packing bins from {packing_bins_path}")
      with open(packing_bins_path, 'r') as f:
        self.bins = json.load(f)
        if sum(len(bin) for bin in self.bins) == len(num_tokens):
          return self.bins
        print("Packing bins do not match the number of tokens. Repacking dataset.")
    else:
      print("Packing arguments have changed or packing bins do not exist. Repacking dataset.")
    self.bins = list()
    for slice_start in trange(0, len(num_tokens), 20000, disable=not DEBUG):
      slice_end = min(slice_start + 20000, len(num_tokens))
      slice_bins = (fast_best_fit_decreasing(
        num_tokens[slice_start:slice_end],
        bin_capacity=bin_capacity,
      ))
      slice_bins = [[slice_start + i for i in slice_bin] for slice_bin in slice_bins]
      self.bins.extend(slice_bins)
    with open(packing_bins_path, 'w') as f:  
      f.write(
        json.dumps(self.bins, indent=2, ensure_ascii=False))
    return self.bins
