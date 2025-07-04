import logging
import random
import subprocess


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

  @staticmethod
  def _make_conversation(item, media_dir, use_cft):
    conversation = list()
    if use_cft:
      raise NotImplementedError()

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


def verify_video(item, media_dir):
  video_path = media_dir / item['video']
  return video_path.exists()


logger = logging.getLogger(__name__)


def reencode(item, media_dir):
  output_path = media_dir.parent / 'vid_reencoded' / item['video']
  video_path = media_dir / item['video']
  if not video_path.exists():
    item['status'] = 'DNE'
    return item

  cmd = [
      "ffmpeg",
      "-y",
      "-i", str(video_path),
      "-c:v", "libx264",
      "-preset", "veryfast",
      "-crf", "23",
      "-c:a", "copy",
      str(output_path)
  ]
  try:
    result = subprocess.run(cmd, stderr=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    if result.returncode == 0:
      logger.info(f"✅ Successfully re-encoded: {video_path}")
    else:
      logger.warning(f"❌ Failed to re-encode: {video_path}")
      logger.debug(result.stderr)
      item['status'] = 'Corrupted'
      return item

  except Exception as e:
    logger.error(f"🚨 Error processing {video_path}: {e}")
    item['status'] = 'PyERROR'
    return item

  item['status'] = 'OK'
  return item
