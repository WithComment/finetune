import datasets

from .utils import filter_image
from .benchmark import BenchmarkDataset

class FashionMnistDataset(BenchmarkDataset):
  options = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
  ]
  
  @staticmethod
  def _get_content(item, media_dir):
    texts = []
    images = [item['image']]
    videos = []
    return texts, images, videos
  
  @staticmethod
  def _make_conversation(item, media_dir, use_cft):
    option_text = ''
    for i, option in enumerate(FashionMnistDataset.options):
      option_text += f"{i}. {option}\n"
    option_text += "Please select the correct option by its number. \n"
    
    conversation = list()
    conversation.append({
      'role': 'user',
      'content': [
        {
          'type': 'image',
          'image': item['image']
        },
        {
          'type': 'text',
          'text': f"Identify the clothing item in the image. Choose from the following options: {', '.join(FashionMnistDataset.options)}"
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