import json
from pathlib import Path

import datasets

from qwenvl.data.benchmark import Benchmark
from qwenvl.data.utils import filter_image

class VQADataset(Benchmark):
  
  add_labels: bool = False
  add_generation_prompt: bool = True
  open_ended: bool = True
  
  @staticmethod
  def _make_conversation(item, media_dir, use_cft=False):
    conversation = list()
    if use_cft:
      raise NotImplementedError("CFT is not implemented for VQADataset.")
    
    restriction_prompt = "Answer straightforwardly and concisely."
    conversation.append({
      'role': 'user',
      'content': [
        {
          'type': 'image',
          'media': item['image']
        },
        {
          'type': 'text',
          'text': item['question']
        },
        {
          'type': 'text',
          'text': restriction_prompt
        }
      ]
    })
    
    return conversation
  
  @staticmethod
  def filter_image(item):
    return filter_image(item, 'image')

  @classmethod
  def download(
      cls,
      ds_key: str = None,
      force: bool = False,
  ):
    if not ds_key:
      raise ValueError("ds_key must be provided for VQADataset type.")
    
    with open(Path(__file__).parent / 'datasets.json', 'r') as f:
      ds_configs = json.load(f)

    ds_config = ds_configs[ds_key]
    dataset_dir = Path(ds_config['dataset_dir'])
    if Path(dataset_dir).exists() and not force:
      print(f"Dataset {dataset_dir} already exists. Use force=True to overwrite.")
      return

    ds = (
      datasets.load_dataset(ds_key)
      .cast_column('image', datasets.Image(decode=False))
      .filter(cls.filter_image, num_proc=32)
      .cast_column('image', datasets.Image(decode=True))
    )
    ds.save_to_disk(dataset_dir)
    return ds
  
  @staticmethod
  def get_content(item):
    """
    Extract content from the item.
    Return a tuple of (texts, images, videos).
    """
    texts = [item['question']]
    images = [item['image']]
    videos = list()
    
    return texts, images, videos

if __name__ == "__main__":
  VQADataset.download(ds_key='flaviagiammarino/vqa-rad')
  VQADataset.download(ds_key='flaviagiammarino/path-vqa')
