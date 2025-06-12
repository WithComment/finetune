import json
from typing import Callable
import datasets
from transformers import AutoTokenizer, Qwen2_5_VLProcessor
from ..argument import ProcessingArguments, DataArguments
from .packing import fast_best_fit_decreasing
from .utils import get_image, get_video_frames
from .base import BaseDataset
import pickle

from ..utils import get_logger
logger = get_logger = get_logger(__name__)

class SFTDataset(BaseDataset):

  for_training: bool = True

  def __init__(
      self,
      name: str,
      processor: Qwen2_5_VLProcessor,
      proc_args: ProcessingArguments,
      data_args: DataArguments,
      force: bool = False
  ) -> None:
    super().__init__(name, processor, proc_args, data_args, force)

    logger.info(f"{self.need_num_content_tokens()=}")
    if self.need_num_content_tokens():
      logger.info("Need to count content tokens.")
      self.get_num_content_tokens()

    bin_capacity = data_args.model_max_length
    ds_w_num_tokens = self.get_num_tokens()
    too_long = ds_w_num_tokens.filter(
      lambda x: x['num_tokens'] > bin_capacity,
      num_proc=BaseDataset.num_proc,
      desc="Filtering too long items"
    )
    logger.info(f"Found {len(too_long)} / {len(self.ds)} ({len(too_long) / len(self.ds):.4f}) items with more than {bin_capacity} tokens.")
    
    if data_args.data_packing:
      logger.info("Data packing is enabled.")
      self.bin_pkl_path = self.ds_dir / 'bins.pkl'
      self.load_bins(
        ds_w_num_tokens['num_tokens'], self.bin_pkl_path, bin_capacity
      )


  def load_bins(self, num_tokens, path, bin_capacity):
    if not path.exists():
      logger.info(f"Creating bins to {path}.")
      self.bins = fast_best_fit_decreasing(
        num_tokens, bin_capacity
      )
      with open(path, 'wb') as f:
        pickle.dump(self.bins, f)
      logger.info(f"Bins saved to {path}.")
    else:
      logger.info(f"Loading bins from pickle file {path}.")
      with open(path, 'rb') as f:
        self.bins = pickle.load(f)


  def need_num_content_tokens(self) -> bool:
    """
    Returns `True` if
    - the dataset does not have the num_content_tokens field,
    - a proc_args.json file is not saved with the dataset,
    - the old proc_args does not match the current proc_args.
    """
    if (
        'num_content_tokens' not in self.ds.features
        or 'media_count' not in self.ds.features
    ):
      return True
    proc_args_path = self.ds_dir / 'proc_args.json'
    if not proc_args_path.exists():
      return True

    with open(proc_args_path, 'r') as f:
      og_proc_args = ProcessingArguments(**json.load(f))

    if og_proc_args != self.proc_args:
      return True

    return False


  @staticmethod
  def _get_num_content_tokens(
      ds: datasets.Dataset,
      processor: Qwen2_5_VLProcessor,
      proc_args: ProcessingArguments,
      get_content_fn: Callable
  ):
    """
    Get the number of content tokens.
    Content are defined by the `get_content` method.
    """
    def _get_num_content_tokens(item):
      texts, images, videos = get_content_fn(item)

      num_tokens = 0
      for text in texts:
        num_tokens += len(processor.tokenizer.encode(text))

      for img in images:
        img = get_image(img)
        result = processor.image_processor(img)
        num_tokens += result['image_grid_thw'].prod() // 4

      for vid in videos:
        vid, fps = get_video_frames(vid, proc_args)
        result = processor.video_processor(vid, fps=fps)
        num_tokens += result['video_grid_thw'].prod() // 4

      item['media_count'] = len(images) + len(videos)
      item['num_content_tokens'] = num_tokens
      return item

    return ds.map(
      _get_num_content_tokens,
      num_proc=BaseDataset.num_proc,
      desc="Counting content tokens",
    )

  def get_num_content_tokens(self):
    """
    Operate on all splits.
    Get the number of content tokens for each item in the dataset.
    The dataset source will always be HF_HOME.
    Save the field num_content_tokens to the dataset,
    along with the processing arguments that affect the content token count.
    """
    self.ds = datasets.load_dataset(self.ds_key)
    self.preprocess()
    self.ds = self._get_num_content_tokens(
      self.ds,
      self.processor,
      self.proc_args,
      self.get_content
    )
    proc_args_path = self.ds_dir / 'proc_args.json'
    with open(proc_args_path, 'w') as f:
      json.dump(self.proc_args.__dict__, f, indent=2)
    self.ds.save_to_disk(self.ds_dir)
    self.ds = self.ds[self.split]
    return self.ds

  def get_num_tokens(self):

    tokenizer = self.processor.tokenizer
    def _get_num_tokens(item):
      tokens = tokenizer.apply_chat_template(
        self._make_conversation(item, self.media_dir, self.use_cft))
      media_count = item['media_count']
      item['num_tokens'] = len(tokens) - media_count + item['num_content_tokens']
      return item

    return self.ds.map(
        _get_num_tokens,
        num_proc=BaseDataset.num_proc,
        desc="Counting total tokens",
    )
