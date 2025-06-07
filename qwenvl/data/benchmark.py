
from abc import ABC, abstractmethod
from pathlib import Path

import datasets
from torch.utils.data import Dataset

from qwenvl.data.generic_dataset import GenericDataset

class Benchmark(GenericDataset, ABC):
  ds_key: str
  add_generation_prompt: bool = True
  add_labels: bool = False
  generation_config: dict
  open_ended: bool
  
  @property
  def result_dir(self) -> Path:
    return self.dataset_dir / 'results'