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
CFT_PROMPTS = [
    # 1. Application of Knowledge (from Situated Learning Theory)
    """Consider the clinical application of the findings in this image. Based on what you see, describe the most likely next step in patient management or diagnosis.""",

    # 2. In-Depth Exploration (from Levels of Processing Theory)
    """Perform an in-depth exploration of this image. Go beyond the primary finding and describe the subtle visual details and nuanced characteristics of the anatomy and any pathologies present.""",

    # 3. Reflective Thinking (from Reflective Practice Theory)
    """Reflect on the diagnostic information presented in this image. Describe the level of certainty for the primary finding. What alternative diagnoses (differentials) could also be considered, even if less likely, based on the visual evidence?""",

    # 4. Creative Interpretation (from Divergent Thinking)
    """Engage creatively with this image. Beyond its immediate diagnostic purpose, what makes this a particularly good or interesting example for teaching, research, or developing new imaging techniques? Highlight its unique visual qualities.""",

    # 5. Summarization and Synthesis (from Generative Learning)
    """Synthesize all the visual information in this image into a single, concise concluding statement or 'impression'. This summary should encapsulate the most critical findings and their overall significance.""",

    # 6. Focus on Key Concepts (from Cognitive Load Theory)
    """Focus on the key concepts presented visually. Identify and describe only the essential, 'hallmark' features in this image that define the primary diagnosis. Disregard incidental or non-contributory findings.""",

    # 7. Contextual Understanding (from Piaget's Theory)
    """Place this image in a broader anatomical and clinical context. Describe how the findings you see relate to the three-dimensional structure of the organ and the typical progression of the suspected condition.""",

    # 8. Critical Analysis (from Bloom's Taxonomy)
    """Critically analyze this image not just for its content, but for its quality. Evaluate potential artifacts, limitations, or technical factors (e.g., motion, noise, poor contrast) that could impact diagnostic interpretation. State whether the image quality is adequate for a confident diagnosis.""",

    # 9. Question-Based Learning (from Paul & Elder's Critical Thinking)
    """Answer the following questions based on the image: 1. What is the primary finding? 2. Where is it located? 3. What are its key visual characteristics? 4. What is the most likely diagnosis? 5. Are there any important secondary findings?""",

    # 10. Comparative Learning (from Relational Frame Theory)
    """Perform a comparative analysis. How are the visual features in this image typical for the primary diagnosis? In what ways do they differ from the classic presentation of a key differential diagnosis?"""
]

class OpenpmcDataset(SFTDataset):
  """
  OpenPMC dataset for training.
  """

  @staticmethod
  def _get_content(item, media_dir):
    texts = [item['sub_caption']]
    images = [item['image']]
    videos = list()

    return texts, images, videos


  @staticmethod
  def _make_conversation(item, media_dir, use_cot, use_cft):
    conversation = list()
    if use_cot:
      raise NotImplementedError()

    conversation.append({
      'role': 'user',
      'content': [
        {
          'type': 'text',
          'text': random.choice(CFT_PROMPTS)
        },
        {
          'type': 'image',
          'image': item['image']
        },
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