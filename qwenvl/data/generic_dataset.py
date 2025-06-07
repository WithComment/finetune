from abc import ABC, abstractmethod
import json
from pathlib import Path
import datasets
from torch.utils.data import Dataset

from qwenvl.data.utils import make_model_input

class GenericDataset(Dataset, ABC):
  ds_key: str
  add_labels: bool
  add_generation_prompt: bool
  generation_config: dict
  dataset_dir: Path
  media_dir: Path | None

  @staticmethod
  def get_dataset_dir(ds_key: str) -> tuple[Path]:
    """
    Get the dataset directory for the given dataset key.
    """
    with open(Path(__file__).parent / 'datasets.json', 'r') as f:
      ds_configs = json.load(f)
    ds_config = ds_configs[ds_key]
    ds_dir = Path(ds_config['dataset_dir'])
    media_dir = ds_config.get('media_dir')
    if media_dir:
      media_dir = Path(media_dir)
    return ds_dir, media_dir
  
  def __len__(self):
    if self.bins:
      return len(self.bins)
    return len(self.ds)


  def __getitem__(self, idx):
    if not self.bins:
      return [self.make_conversation(self.ds[idx])]
    idx = self.bins[idx]
    items = self.ds.select(idx)
    conversation = list()
    for item in items:
      conversation.extend(self.make_conversation(item))
    return conversation


  def make_model_input(self, conversations: list[dict]) -> dict:
    """
    Convert a batch of human readable input from self[i] to inputs ready for the model.
    E.g., input_ids, attention_mask, image_pixel_values, etc.
    """
    return make_model_input(
      conversations,
      processor=self.processor,
      proc_args=self.proc_args,
      media_dir=self.media_dir,
      add_labels=self.add_labels
    )

  @staticmethod
  @abstractmethod
  def get_content(item):
    """
    Extract content from the item.
    Return a tuple of (texts, images, videos).
    """
    raise NotImplementedError("Subclasses must implement this method.")


  @staticmethod
  @abstractmethod
  def _make_conversation(item, media_dir, use_cft):
    """
    Create a conversation from the item.
    E.g.
    [
      {
        'role': 'user',
        'content': [
          {'type': 'text', 'text': some text},
          {'type': 'image', 'image': path to image or Image object}
        ]
      },
      {
        'role': 'assistant',
        'content': [{'type': 'text', 'text': some text}]
      }
    ]
    """
    raise NotImplementedError("Subclasses must implement this method.")


  @classmethod
  @abstractmethod
  def download(cls, ds_key, force) -> datasets.DatasetDict:
    """
    Prepares the dataset up until steps that
    depend on training/inference parameters.
    I.e., after this step, the dataset should locate at the directory pointed by
    dataset_dir of the datasets.json file.
    And the dataset should be ready to create number of tokens.
    This method should be implemented by subclasses.
    """
    raise NotImplementedError("Subclasses must implement this method.")
