import datasets
from . import BenchmarkDataset
from .utils import filter_image

class VQADataset(BenchmarkDataset):
  
  @staticmethod
  def _get_content(item, media_dir):
    texts = [item['question']]
    images = [item['image']]
    videos = list()
    return texts, images, videos
  
  @staticmethod
  def _make_conversation(item, media_dir, use_cft):
    conversation = list()
    restriction_prompt = "Answer straightforwardly and concisely: "
    conversation.append({
      'role': 'user',
      'content': [
        {
          'type': 'image',
          'image': item['image']
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
  def _preprocess(ds, media_dir, num_proc) -> datasets.DatasetDict:
    ds = BenchmarkDataset.add_ids(ds)
    
    def _filter_image(item):
      """
      Filter out items without an image.
      """
      return filter_image(item, 'image', 'id')
    
    return (ds
      .cast_column('image', datasets.Image(decode=False))
      .filter(_filter_image, num_proc=num_proc)
      .cast_column('image', datasets.Image(decode=True))
    )
