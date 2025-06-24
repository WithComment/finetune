from io import BytesIO
import json
import os
from pathlib import Path
import random

import datasets
import PIL.Image as PILImage

from .sft import SFTDataset
from .utils import filter_image


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
  OpenPMC dataset for training.
  """

  @staticmethod
  def _get_content(item, media_dir):
    texts = [item['intext_refs_summary'] or item['intext_refs']]
    texts.append(item['sub_caption'])
    images = [item['image']]
    videos = list()

    return texts, images, videos


  @staticmethod
  def _make_conversation(item, media_dir, use_cot):
    conversation = list()
    if use_cot:
      raise NotImplementedError()

    if not (intext_ref := item['intext_refs_summary']):
      intext_ref = item['intext_refs']
    conversation.append({
      'role': 'user',
      'content': [
        {
          'type': 'image',
          'image': item['image']
        },
        {
          'type': 'text',
          'text': random.choice(IMG_PROMPTS)
        },
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
  def _preprocess(ds, media_dir, num_proc) -> datasets.DatasetDict:
    
    def _filter_image(item):
      """
      Filter out items without an image.
      """
      return filter_image(item, 'jpg', '__key__')
    
    def _load_jsonl(item):
      return json.loads(item['jsonl'])

    return (ds
      .map(_load_jsonl, num_proc=num_proc, remove_columns=['jsonl'], desc='loading json')
      .cast_column('jpg', datasets.Image(decode=False))
      .filter(_filter_image, num_proc=num_proc)
      .cast_column('jpg', datasets.Image(decode=True))
      .remove_columns(['image'])
      .rename_column('jpg', 'image')
    )

if __name__ == "__main__":
  pass