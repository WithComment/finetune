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
  ds_dir: Path
  media_dir: Path | None
  NUM_PROC = 32

  @staticmethod
  def get_ds_config(name: str) -> dict[str, Path | str]:
    """
    Get the dataset config for the given dataset name.
    """
    with open(Path(__file__).parent / 'datasets.json', 'r') as f:
      ds_configs = json.load(f)
    ds_config = ds_configs[name]
    ds_config['ds_dir'] = Path(ds_config['ds_dir'])
    if ds_config['media_dir'] is not None:
      ds_config['media_dir'] = Path(ds_config['media_dir'])
    
    return ds_config


  def __len__(self):
    if self.bins:
      return len(self.bins)
    return len(self.ds)


  def __getitem__(self, idx):
    if not self.bins:
      return self.make_conversation(self.ds[idx])
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


  def drop_non_json_fields(self, item: dict) -> dict:
    """
    Keep JSON serializable fields from the item.
    """
    return {k: v for k, v in item.items() if isinstance(v, (str, int, float, bool, list, dict))}


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
  
  
  @abstractmethod
  def preprocess(self) -> datasets.DatasetDict:
    """
    Preprocess the dataset, so it is ready for `make_conversation`.
    """
    raise NotImplementedError()


  def download_and_process(self, name) -> datasets.DatasetDict:
    """
    Download the dataset to HF_HOME.
    Pre process the dataset.
    And save to dataset_dir.
    """
    ds_config = self.get_ds_config(name)
    self.ds = datasets.load_dataset(ds_config['ds_key'])
    self.ds = self.preprocess(ds_config)
    self.ds.save_to_disk(ds_config['ds_dir'])
    return self.ds
