
from abc import ABC, abstractmethod
import copy
from functools import partial
from torch.utils.data import Dataset
from transformers import AutoProcessor

from qwenvl.data.packing import fast_best_fit_decreasing
from qwenvl.data.utils import get_num_content_tokens, get_num_tokens
from qwenvl.train.argument import ProcessingArguments


class SFTDataset(Dataset, ABC):
  """
  OpenPMC dataset for training and evaluation.
  """
  def __init__(
      self,
      media_dir,
      processor: AutoProcessor,
      proc_args: ProcessingArguments,
      use_cft=False,
      dataset_dir=None,
      pack=False
  ):
    super(Dataset, self).__init__()
    self.use_cft = use_cft
    self.media_dir = media_dir
    self.processor = copy.deepcopy(processor)
    self.ds = get_num_content_tokens(
        dataset_dir,
        media_dir,
        processor=self.processor,
        proc_args=proc_args,
        get_content_fn=self.get_content,
        num_proc=32,
    )
    self.tokenizer = self.processor.tokenizer
    self._make_conversation = partial(
      self.make_conversation,
      media_dir=self.media_dir,
      use_cft=self.use_cft
    )
    if not pack:
      self.groups = [(i,) for i in range(len(self.ds))]
    else:
      self.groups = fast_best_fit_decreasing(get_num_tokens(
          self.ds,
          self._make_conversation,
          tokenizer=self.tokenizer,
        ),
        proc_args.model_max_length
      )

  def __len__(self):
    return len(self.groups)


  def __getitem__(self, idx):
    idx = self.groups[idx]
    items = self.ds.select(idx)
    conversation = list()
    for item in items:
      conversation.extend(self._make_conversation(item))
    return conversation
  
  
  def validate_media(self):
    """
    Validate media files in the item.
    """
    return 

  
  @staticmethod
  @abstractmethod
  def get_content(item):
    """
    Extract content from the item.
    This method should be implemented by subclasses.
    """
    raise NotImplementedError("Subclasses must implement this method.")
  

  @staticmethod
  @abstractmethod
  def make_conversation(item, media_dir, use_cft):
    """
    Create a conversation from the item.
    This method should be implemented by subclasses.
    """
    raise NotImplementedError("Subclasses must implement this method.")
  
  @staticmethod
  @abstractmethod
  def preprocess(splits):
    """
    Preprocess the dataset, i.e., adding removing columns, filtering, etc.
    This method should be implemented by subclasses.
    """
    raise NotImplementedError("Subclasses must implement this method.")