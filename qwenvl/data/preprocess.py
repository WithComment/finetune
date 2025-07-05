from abc import ABC, abstractmethod
from pathlib import Path
import pickle
import shutil
from typing import Callable

import datasets

from qwenvl.argument import ProcessingArguments
from qwenvl.data.conversation import ConversationMaker
from qwenvl.data.input_processor import InputProcessor
from qwenvl.data.packing import fast_best_fit_decreasing
from qwenvl.data.utils import get_image, get_video_frames, smart_resize
from qwenvl.utils import get_logger

logger = get_logger(__name__)

class PreprocessStrategy(ABC):
  num_proc: int = 32
  def __init__(self, num_proc: int = None):
    if num_proc is not None:
      self.num_proc = num_proc
      
      
  @abstractmethod
  def __call__(self, ds: datasets.Dataset):
    pass


class GetNumMediaTokensStrategy(PreprocessStrategy):
  
  def __init__(self, get_content_fn: Callable, config: ProcessingArguments, **kwargs):
    super().__init__(**kwargs)
    self.get_content_fn = get_content_fn
    self.config = config

  def get_num_content_tokens(self, item):
    try:
      texts, images, videos = self.get_content_fn(item)

      num_tokens = 0
      
      for img in images:
        img = get_image(img)
        h, w, h_tokens, w_tokens = smart_resize(
          img.height, img.width,
          self.config.image_max_pixels,
          self.config.image_min_pixels,
        )
        num_tokens += h_tokens * w_tokens

      for vid in videos:
        vid, fps = get_video_frames(vid, self.config)
        nframes = vid.shape[0]
        frame = vid[:1]
        h, w, h_tokens, w_tokens = smart_resize(
          frame.shape[-2], frame.shape[-1],
          self.config.video_max_pixels,
          self.config.video_min_pixels,
        )
        num_tokens += h_tokens * w_tokens * nframes // self.config.temporal_patch_size
    except Exception as e:
      logger.error(e)
      texts, images, videos = [], []
      num_tokens = 0

    item['num_media'] = len(images) + len(videos)
    item['num_media_tokens'] = num_tokens
    return item

  def __call__(self, ds: datasets.Dataset):
    """
    Get the number of content tokens.
    Content are defined by the `get_content` method.
    """

    return ds.map(
      self.get_num_content_tokens,
      num_proc=self.num_proc,
      desc="Counting media tokens",
    )


class GetNumTokensStrategy(PreprocessStrategy):
  def __init__(
    self,
    cm: ConversationMaker,
    processor: InputProcessor,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.cm = cm
    self.processor = processor
    
  def get_num_tokens(self, item):
    tokens = self.processor.get_text_length(self.cm(item))
    num_media = item['num_media']
    item['num_tokens'] = len(tokens) - num_media + item['num_media_tokens']
    return item
  
  def __call__(self, ds: datasets.Dataset):

    return ds.map(
        self.get_num_tokens,
        num_proc=self.num_proc,
        desc="Counting total tokens",
    )



class FilterStrategy(PreprocessStrategy):
  def __init__(self, filter_fns: Callable[[dict], bool] | list[Callable[[dict], bool]], **kwargs):
    super().__init__(**kwargs)
    if not isinstance(filter_fns, list):
      filter_fns = [filter_fns]
    self.filter_fns = filter_fns

  def preprocess(self, ds: datasets.Dataset) -> datasets.Dataset:
    init_len = len(ds)
    for fn in self.filter_fns:
      total = len(ds)
      ds = ds.filter(
        fn,
        num_proc=self.num_proc,
        desc=f"Filtering dataset with {fn.__name__}",
      )
      removed = total - len(ds)
      logger.info(f"{fn.__name__} filtered out {removed} ({100 * removed / total:.2f}%) items from the dataset.")
    
    removed = init_len - len(ds)
    logger.info(f"Total filtered out {removed} ({100 * removed / init_len:.2f}%) items from the dataset.")
    return ds


class SaveStrategy(PreprocessStrategy):
  def __init__(self, save_path: str | Path, exists_ok: bool = True, **kwargs):
    super().__init__(**kwargs)
    self.save_path = Path(save_path)
    self.temp_path = self.save_path.with_name(f"{self.save_path.name}_tmp")
    self.exists_ok = exists_ok

  def __call__(self, ds: datasets.Dataset):
    logger.info(f"Saving dataset to {self.save_path}.")
    if self.save_path.exists() and not self.exists_ok:
      raise FileExistsError(f"Dataset already exists at {self.save_path}. Use `exists_ok=True` to overwrite.")
    self.save_path.mkdir(parents=True, exist_ok=True)
    if self.save_path.exists():
      ds.save_to_disk(self.temp_path)
      logger.info(f"Temporary dataset saved to {self.temp_path}.")
      shutil.rmtree(self.save_path)
      logger.info(f"Removed existing dataset at {self.save_path}.")
      shutil.move(self.temp_path, self.save_path)
    else:
      ds.save_to_disk(self.save_path)
    logger.info(f"Dataset saved to {self.save_path}.")
    return ds
  

def pack_dataset(ds: datasets.Dataset, bin_capacity: int):
  if 'num_tokens' not in ds.features:
    raise ValueError("Dataset must have 'num_tokens' field for packing.")
  
  logger.info("Packing dataset into bins.")
  bins = fast_best_fit_decreasing(ds['num_tokens'], bin_capacity)
  logger.info(f"Packed dataset into {len(bins)} bins.")
  return bins
