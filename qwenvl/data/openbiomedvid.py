import logging
import random
import subprocess

from qwenvl.data.utils import make_cot, verify_video


from .sft import SFTDataset


VID_PROMPTS = (
    "Please describe the biomedical content shown in this video.",
    "What medical or clinical content can you observe in this video?",
    "Could you explain the medical aspects shown in this footage?",
    "Please provide a description of the medical content demonstrated in this video.",
    "What biomedical information is being presented in this video?",
    "Can you describe the medical content shown in this footage?",
    "Please explain what you observe in this medical video.",
    "What medical or clinical elements are demonstrated in this video?",
    "Could you describe the biomedical content presented here?",
    "Please detail the medical information shown in this video.",
    "What do you observe in this medical footage?",
    "Can you explain the biomedical content demonstrated here?",
    "Please describe what's being shown in this medical video.",
    "What medical content is being presented in this footage?",
    "Could you detail the biomedical aspects shown in this video?",
    "Please explain the medical elements demonstrated here.",
    "What clinical or medical content do you observe in this video?",
    "Can you describe the biomedical information shown in this footage?",
    "Please provide an explanation of the medical content in this video.",
    "What medical or clinical aspects are being demonstrated here?"
)


class OpenbiomedvidDataset(SFTDataset):
  """
  Openbiomedvid dataset for training and evaluation.
  """

  @staticmethod
  def _get_content(item, media_dir):
    texts = [item['caption']]
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
          use_cot=self.use_cot
        )
      )
    return conversation

  @staticmethod
  def _make_conversation(item, subtitle_dir, media_dir, use_cot):
    conversation = list()
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
                'text': random.choice(VID_PROMPTS)
            },
        ]
    })
    conversation.append({
        'role': 'assistant',
        'content': [
            {
                'type': 'text',
                'text': cot
            },
            {
                'type': 'text',
                'text': item['caption']
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


logger = logging.getLogger(__name__)
