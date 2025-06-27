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
  def _make_conversation(item, media_dir, mode):
    raise NotImplementedError("VQADataset does not support _make_conversation method. Use make_conversation instead.")
  
  def make_conversation(self, bin):
    for item in bin:
      conversation = list()
      match self.sys_prompt:
        case 'default':
          sys_prompt = "You are a helpful assistant."
        case 'custom':
          sys_prompt = ("You are a **question answering** assistant. You task is to **answer the question** based on the provided image. "
                      "You should **not** provide any additional information or context beyond the image and the question.")
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
            'type': 'image',
            'image': item['image']
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
