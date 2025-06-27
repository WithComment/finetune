# Dataset processing workflow

# 1. Download the dataset to HF_HOME. This will not be touched.
# 2. Preprocess the dataset, save it to ds_dir. This will serve as the source. Should not be touched.
# 3. For each instance of the dataset class, we need to count the content tokens, i.e., the amount of tokens not part of the instruction prompts.
# 4. Save the dataset with counted tokens, along with the arguments that affect the content token count, to some directory.

# Considerations:
# All arguments that affect the content token count should be grouped.
# No other arguments should be in that argument group.
# use_cft, add_generation_prompt should not be part of persistent arguments.
# Their values should be passed to the make_conversation method.
# Packing takes negligible time, so we can afford repacking the dataset every time an instance is created.
# Since it is possible for two different datasets to share the same structure, e.g., path-vqa and vqa-rad, ds_key, ds_dir, media_dir should be instance attributes.
# make_conversation method should


import json
from pathlib import Path
import shutil
import datasets
import torch
from torch.utils.data import Dataset
from transformers import Qwen2_5_VLProcessor
from abc import ABC, abstractmethod

from ..argument import DataArguments, ProcessingArguments
from .utils import make_model_input
from ..utils import get_logger
logger = get_logger(__name__)

torch.set_num_threads(1)  # Ensure single-threaded processing for datasets

class BaseDataset(Dataset, ABC):
  """
  Base class for all datasets and benchmarks.
  """
  name: str
  ds: datasets.Dataset
  ds_config: dict
  ds_dir: Path
  media_dir: Path | None
  ds_key: str
  mode: str
  processor: Qwen2_5_VLProcessor
  proc_args: ProcessingArguments
  for_training: bool
  num_proc: int = 24

  def __init__(
      self,
      name: str,
      processor: Qwen2_5_VLProcessor,
      proc_args: ProcessingArguments,
      data_args: DataArguments,
  ) -> None:
    """
    If the dataset has been downloaded and processed (including content tokens),
    it will be loaded from ds_dir.
    Otherwise, it will be downloaded from HF_HOME and processed.
    """
    self.name = name
    self.processor = processor
    self.proc_args = proc_args
    self.data_args = data_args
    self.mode = data_args.mode
    self.split = data_args.split
    self.sys_prompt = data_args.sys_prompt
    
    # Validate mode
    if self.mode not in ["cft", "cpt", "ift"]:
      raise ValueError(f"Invalid mode: {self.mode}. Must be one of ['cft', 'cpt', 'ift']")
    
    self.ds_config = self._get_ds_config(name)
    self.ds_dir = self.ds_config['ds_dir']
    self.media_dir = self.ds_config['media_dir']
    self.ds_key = self.ds_config['ds_key']
    if data_args.force or not self.ds_dir.exists():
      logger.info(f"{data_args.force=}, {self.ds_dir.exists()=}, (re)downloading dataset {self.ds_key} to {self.ds_dir}")
      shutil.rmtree(self.ds_dir, ignore_errors=True)
      self.ds = datasets.load_dataset(self.ds_key)
      self.ds = self.preprocess()
      self.ds.save_to_disk(str(self.ds_dir))
      self.ds.cleanup_cache_files()
      logger.info(f"Dataset {self.ds_key} saved to {self.ds_dir}")
      self.ds = self.ds[self.split]
      
    else:
      logger.info(f"Loading dataset {self.ds_key} from {self.ds_dir}")
      self.ds = datasets.load_from_disk(str(self.ds_dir))[self.split]
      
    # Default to no packing.
    self.bins = [[i] for i in range(len(self.ds))]
    
  @staticmethod
  def _get_ds_config(name: str) -> dict:
    """
    Get the dataset config for the given dataset name.
    """
    from . import avail_datasets
    ds_config = avail_datasets.get(name)
    if ds_config is None:
      raise ValueError(f"Dataset {name} is not available.")
    ds_config['ds_dir'] = Path(ds_config['ds_dir'])
    if ds_config['media_dir'] is not None:
      ds_config['media_dir'] = Path(ds_config['media_dir'])
    
    ds_config = ds_config
    return ds_config


  @staticmethod
  def add_ids(
      ds: datasets.Dataset | datasets.DatasetDict
  ):
    def _add_ids(item, idx):
      item['id'] = idx
      return item
    return ds.map(
      _add_ids,
      with_indices=True,
      num_proc=BaseDataset.num_proc
    )
    
  @staticmethod
  @abstractmethod
  def _preprocess(
      ds: datasets.DatasetDict,
      media_dir: Path,
      num_proc: int
  ) -> datasets.DatasetDict:
    raise NotImplementedError()


  def preprocess(self) -> datasets.DatasetDict:
    """
    Preprocess all splits of the dataset.
    """
    logger.info(f"Preprocessing dataset {self.name}")
    self.ds = self._preprocess(self.ds, self.media_dir, self.num_proc)
    return self.ds


  @staticmethod
  @abstractmethod
  def _get_content(item: dict, media_dir: Path) -> tuple[list[str], list, list[str]]:
    """
    Extract content from the item.
    Convert media relative paths to absolute paths.
    No additional processing is done.
    """
    raise NotImplementedError()
  
  
  def get_content(self, item: dict):
    return self._get_content(
      item,
      media_dir=self.media_dir
    )
  
  
  @staticmethod
  @abstractmethod
  def _make_conversation(
      row: dict,
      media_dir: Path | None,
      mode: str
  ) -> list[dict]:
    """Always operate on a single row, i.e., a dict.
    """
    raise NotImplementedError()
  

  def make_conversation(
      self,
      bin: list[dict],
  ) -> list[dict]:
    """
    Create a conversation from a bin.
    If a bin contain multiple rows, they will be merged into a single conversation.
    The resulting conversation is self-contained, i.e., 
    all paths to media files are absolute paths.
    Example:
    [
      {
        'role': 'user',
        'content': [{'type': 'text', 'text': '...'}, {'type': 'image', 'image': '...'}]
      },
      {
        'role': 'assistant',
        'content': [{'type': 'text', 'text': '...'}]
      },
      ...
    ]
    """
    conversation = list()
    for item in bin:
      conversation.extend(
        self._make_conversation(
          item,
          media_dir=self.media_dir,
          mode=self.mode
        )
      )
    return conversation
    
  
  def make_model_input(self, batch_convo: list[dict]) -> tuple[dict, str]:
    return make_model_input(
      conversations=batch_convo,
      processor=self.processor,
      proc_args=self.proc_args,
      for_training=self.for_training,
      mode=self.mode
    )


  def collate_fn(self, batch: list[dict] | list[list[dict]]) -> dict:
    """
    Transform a list of rows of the dataset into a dictionary suitable for model input.
    Including input_ids, attention_mask, image_pixel_values, etc.
    """
    if not isinstance(batch[0], list):
      batch = [batch]
    batch_convo = list()
    for pack in batch:
      batch_convo.append(self.make_conversation(pack))
    return self.make_model_input(batch_convo)[0]


  def __len__(self):
    return len(self.bins)
  
  
  @property
  def n_samples(self) -> int:
    return len(self.ds)
  
  @property
  def packed(self) -> bool:
    return isinstance(self.bins[0], list)
  
  def _get_bin(self, idx: int) -> list[dict]:
    return [self.ds[i] for i in self.bins[idx]]

  
  def __getitem__(self, idx: int | slice) -> list[dict] | list[list[dict]]:
    """
    The item is always wrapped in a list to distinguish batch vs packing.
    Return a list of dict if idx is an int.
    Return a list of lists of dicts if idx is a slice.
    """
    if isinstance(idx, int):
      return self._get_bin(idx)
    
    return [
      self._get_bin(i) for i in self.bins[idx]
    ]
  

  @staticmethod
  def drop_non_json_fields(item: dict) -> dict:
    """
    Drop fields that are not JSON serializable.
    """
    return {k: v for k, v in item.items() if isinstance(v, (str, int, float, bool, list, dict))}