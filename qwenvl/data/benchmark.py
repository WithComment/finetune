import json
from typing import Callable
import datasets
from transformers import AutoTokenizer, Qwen2_5_VLProcessor
from ..argument import ProcessingArguments, DataArguments
from .packing import fast_best_fit_decreasing
from .utils import get_image, get_video_frames

from . import BaseDataset

class BenchmarkDataset(BaseDataset):
  
  for_training: bool = False
  
  def __init__(self, name, processor, proc_args, data_args, force = False):
    super().__init__(name, processor, proc_args, data_args, force)