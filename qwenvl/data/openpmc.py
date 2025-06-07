from io import BytesIO
import json
import os
from pathlib import Path
import random

import datasets
import PIL.Image as PILImage

from qwenvl.data.sft_dataset import SFTDataset
from qwenvl.data.utils import filter_image, get_media


IMG_PROMPTS = (
    "Generate a concise and descriptive caption for the provided image.",
    "Describe this image with a short, informative textual caption.",
    "Write a brief, accurate caption for the visual content shown.",
    "Create a suitable caption to accompany this specific image input.",
    "Provide a short textual summary caption reflecting this image's content.",
    "Please generate an appropriate and concise caption for this picture.",
    "Summarize the key visual elements of this image in a caption.",
    "Compose a caption that effectively describes the scene in the image.",
    "Offer a succinct caption detailing the main focus of this visual.",
    "Formulate a fitting and descriptive caption for the image presented.",
)

class OpenpmcDataset(SFTDataset):
  """
  OpenPMC dataset for training and evaluation.
  """

  add_labels: bool = True
  add_generation_prompt: bool = False
  ds_key: str = 'vector-institute/open-pmc'

  @staticmethod
  def get_content(item):
    """
    Extracts text, images, and videos from an OpenPMC item.
    """
    texts = [item['intext_refs_summary'] or item['intext_refs']]
    texts.append(item['sub_caption'])
    images = [item['image']]
    videos = list()

    return texts, images, videos


  @staticmethod
  def _make_conversation(item, media_dir, use_cft=False):
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
          'image': item['image']
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
  
  @staticmethod
  def filter_image(item):
    """
    Filter out items without an image.
    """
    return filter_image(item, 'jpg', '__key__')
  
  @staticmethod
  def load_jsonl(item):
    return json.loads(item['jsonl'])

  @classmethod
  def download(
      cls,
      ds_key: str = None,
      force: bool = False,
      save_to_disk: bool = True,
  ) -> datasets.DatasetDict:
    if ds_key:
      raise RuntimeWarning("ds_key is not supported.")

    ds_key = cls.ds_key

    with open(Path(__file__).parent / 'datasets.json', 'r') as f:
      ds_configs = json.load(f)

    ds_config = ds_configs[ds_key]
    dataset_dir = Path(ds_config['dataset_dir'])
    if Path(dataset_dir).exists() and not force:
      print(f"Dataset {dataset_dir} already exists. Use force=True to overwrite.")
      return datasets.load_from_disk(dataset_dir)
    ds = (
      datasets.load_dataset(ds_key)
      .map(cls.load_jsonl, num_proc=32, remove_columns=['jsonl'])
      .cast_column('jpg', datasets.Image(decode=False))
      .filter(cls.filter_image, num_proc=32)
      .cast_column('jpg', datasets.Image(decode=True))
      .remove_columns(['image'])
      .rename_column('jpg', 'image')
    )
    if save_to_disk:
      ds.save_to_disk(dataset_dir)

    return ds

if __name__ == "__main__":
  OpenpmcDataset.download(force=True)