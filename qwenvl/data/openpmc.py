import json
import os
from pathlib import Path
import random

import datasets
import PIL.Image as PILImage

from qwenvl.data.sft_dataset import SFTDataset
from qwenvl.data.utils import get_media_names
from qwenvl.data import IMG_PROMPTS


class OpenpmcDataset(SFTDataset):
  """
  OpenPMC dataset for training and evaluation.
  """

  @staticmethod
  def get_content(item):
    """
    Extracts text, images, and videos from an OpenPMC item.
    """
    texts = [item['intext_refs_summary'] or item['intext_refs']]
    texts.append(item['sub_caption'])
    images = get_media_names(item, ['image', 'images'])
    videos = get_media_names(item, ['video', 'videos'])
    
    return texts, images, videos


  @staticmethod
  def preprocess(splits):
    with open(Path(__file__).parent / 'datasets.json', 'r') as f:
      ds_configs = json.load(f)

    for split in splits:
      dataset_dir = ds_configs[f'openpmc_{split}']['dataset_dir']
      media_dir = ds_configs[f'openpmc_{split}']['media_dir']
      ds = (
        datasets.load_dataset('vector-institute/open-pmc', split=split)
        .map(load_jsonl, num_proc=32, remove_columns=['jsonl'])
        .remove_columns(['jpg', '__key__', '__url__'])
        .filter(lambda item: filter_invalid(item, media_dir), num_proc=32)
      )
      ds.save_to_disk(dataset_dir)


  @staticmethod
  def make_conversation(item, media_dir, use_cft=False):
    conversation = list()
    if use_cft:
      raise NotImplementedError()

    if not (intext_ref := item['intext_refs_summary']):
      intext_ref = item['intext_refs']
    conversation.append({
      'role': 'user',
      'content': [
        {
          'type': 'text',
          'text': intext_ref
        },
        {
          'type': 'text',
          'text': random.choice(IMG_PROMPTS)
        },
        {
          'type': 'image',
          'image': os.path.join(media_dir, item['image'])
        }
      ]
    })
    conversation.append({
      'role': 'assistant',
      'content': [
        {
          'type': 'text',
          'text': item['sub_caption']
        }
      ]
    })
    return conversation


def load_jsonl(item):
  d = json.loads(item['jsonl'])
  return d


def filter_invalid(item, media_dir):
  try:
    with PILImage.open(str(Path(media_dir) / item['image'])) as img:
      if img.height < 28 or img.width < 28:
        return False
      img.verify()
    return True
  except Exception as e:
    return False


if __name__ == "__main__":
  process_openpmc(['train', 'validation', 'test'])