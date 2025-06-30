from pathlib import Path
import random

import datasets
from qwenvl.data.base import BaseDataset
from qwenvl.data.openbiomedvid import VID_PROMPTS
from qwenvl.data.utils import make_cot, reencode, verify_video
from .benchmark import BenchmarkDataset
from ..utils import get_logger

logger = get_logger(__name__)

class SurgeryVidDataset(BenchmarkDataset):
  
  @staticmethod
  def _get_content(item, media_dir):
    texts = [item['question']]
    images = list()
    videos = [media_dir / item['video']]

    return texts, images, videos
  
  def make_conversation(self, bin):
    conversation = list()

    for item in bin:
      conversation.extend(
        self._make_conversation(
          item,
          subtitle_dir=self.media_dir.with_name('sub_segments'),
          media_dir=self.media_dir,
          use_cot=self.data_args.use_cot
        )
      )
    return conversation

  @staticmethod
  def _make_conversation(item, subtitle_dir, media_dir, use_cot):
    raise NotImplementedError()
  
  def make_conversation(self, bin):
    for item in bin:
      conversation = list()
      match self.sys_prompt:
        case 'default':
          sys_prompt = "You are a helpful assistant."
        case 'custom':
          sys_prompt = ("You are a **question answering** assistant. You task is to **answer the question** based on the provided vision input. "
                      "You should **not** provide any additional information or context beyond the vision input and the question.")
        case _:
          sys_prompt = ''
          
      conversation.append({
        'role': 'system',
        'content': [{
            'type': 'text',
            'text': sys_prompt
          }]
      })
      restriction_prompt = "Answer straightforwardly and concisely: "
      conversation.append({
        'role': 'user',
        'content': [
          {
            'type': 'video',
            'video': self.media_dir / item['video']
          },
          {
            'type': 'text',
            'text': restriction_prompt
          },
          {
            'type': 'text',
            'text': item['question']
          },
        ]
      })
    
    return conversation

  @staticmethod
  def _preprocess(ds, media_dir, num_proc):
    def _verify_video(item):
      return verify_video(item, media_dir)

    ds = ds.filter(
        _verify_video,
        num_proc=num_proc,
        desc="Filtering out items with missing videos"
    )
    return ds


if __name__ == "__main__":
  from . import avail_datasets
  
  ds = datasets.load_dataset("connectthapa84/SurgeryVideoQA", split='test')
  ds_config = avail_datasets['surgery-vid']
  video_set = set()
  
  def _filter_duplicate(item):
    if item['video'] in video_set:
      return False
    video_set.add(item['video'])
    return True
  ds = ds.filter(_filter_duplicate, num_proc=BaseDataset.num_proc).map(
    reencode,
    fn_kwargs={
      'media_dir': Path(ds_config['media_dir']),
      'logger': logger
    },
  )
  st_count = dict()
  for st in ds.unique('status'):
    st_count[st] = len(ds.filter(lambda x: x['status'] == st), num_proc=BaseDataset.num_proc)
  for st, count in st_count.items():
    logger.info(f"Status: {st}, Count: {count}")
  logger.info(f"Total items: {len(ds)}")