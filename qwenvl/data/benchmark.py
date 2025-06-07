
from abc import ABC, abstractmethod
import copy
from pathlib import Path

import datasets
from torch.utils.data import Dataset
from transformers import AutoProcessor

from qwenvl.data.generic_dataset import GenericDataset
from qwenvl.train.argument import DataArguments, ProcessingArguments

class Benchmark(GenericDataset, ABC):
  ds_key: str
  add_generation_prompt: bool = True
  add_labels: bool = False
  bins: None
  generation_config: dict
  open_ended: bool
  
  def __init__(
      self,
      processor: AutoProcessor,
      data_args: DataArguments,
      proc_args: ProcessingArguments,
      sampling_rate: float = 1.0,
      ds_key: str = None,
      dataset_dir: Path = None,
      media_dir: Path = None,
  ):
    super(Dataset, self).__init__()
    self.ds_key = ds_key
    self.processor = copy.deepcopy(processor)
    self.proc_args = copy.deepcopy(proc_args)
    
    self.dataset_dir, self.media_dir = self.get_dataset_dir(self.ds_key)
    if not self.dataset_dir.exists():
      self.ds = self.download(ds_key, force=False, save_to_disk=False)[data_args.split]
    else:
      self.ds = datasets.load_from_disk(self.dataset_dir)[data_args.split]
    self.bins = None
