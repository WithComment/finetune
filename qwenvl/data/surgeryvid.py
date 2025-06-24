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
    conversation = list()
    restriction_prompt = "Answer concisely in no more than a few words: "
    cot = None
    if use_cot:
      cot = subtitle_dir / item['video'].replace('.mp4', '.en.vtt')
      if not cot.exists():
        logger.warning(f"Subtitle file {cot} does not exist, skipping COT.")
        cot = None
      else:
        cot = make_cot(cot)
    conversation.append({
        'role': 'user',
        'content': [
            {
                'type': 'video',
                'video': str(media_dir / item['video'])
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
    if cot:
      conversation.append({
          'role': 'assistant',
          'content': [
              {
                  'type': 'text',
                  'text': cot
              }
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