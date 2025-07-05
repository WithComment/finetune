import transformers
from dataclasses import dataclass, field
from typing import Optional

PATCH_WIDTH = 28
PATCH_SIZE = PATCH_WIDTH ** 2


@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
      default="Qwen/Qwen2.5-VL-3B-Instruct")
  tune_mm_llm: bool = field(default=True)
  tune_mm_mlp: bool = field(default=True)
  tune_mm_vision: bool = field(default=False) 


@dataclass
class DataArguments:
  dataset_use: str = field(default="")
  packing: bool = field(default=False)
  split: str = field(default="train")
  model_max_length: int = field(default=16384)
  portion: float = field(default=1.0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
  cache_dir: Optional[str] = field(default=None)
  optim: str = field(default="adamw_bnb_8bit")
  mm_projector_lr: Optional[float] = None
  vision_tower_lr: Optional[float] = None
  min_lr_ratio: Optional[float] = field(default=0.1, metadata={"help": "Minimum learning rate ratio for cosine_with_min_lr scheduler"})


@dataclass
class ProcessingArguments:
  """
  These arguments are important for content token count.
  They will be saved.
  """
  image_min_pixels: int = field(default=PATCH_SIZE * 64)
  image_max_pixels: int = field(default=PATCH_SIZE * 2048)
  default_to_square: bool = field(default=False)

  video_min_pixels: int = field(default=PATCH_SIZE * 64)
  video_max_pixels: int = field(default=PATCH_SIZE * 512)
  video_max_frames: int = field(default=420)
  video_min_frames: int = field(default=4)
  base_interval: int = field(default=1)
  temporal_patch_size: int = field(default=2, metadata={"help": "Temporal patch size for video processing"})
  
  sys_prompt: str = field(default='')
  cft_prompt: str = field(default='')
  use_chat_template: bool = field(default=False, metadata={"help": "Use chat template for text input"})
  add_generation_prompt: bool = field(default=False, metadata={"help": "Add generation prompt to text input"})
  add_vision_id: bool = field(default=False, metadata={"help": "Add vision id to text input"})
  ignore_idx: int = field(default=-100, metadata={"help": "Index to ignore in loss calculation"})
  
  @property
  def media_params(self):
    return {
        "image_min_pixels": self.image_min_pixels,
        "image_max_pixels": self.image_max_pixels,
        "default_to_square": self.default_to_square,
        "video_min_pixels": self.video_min_pixels,
        "video_max_pixels": self.video_max_pixels,
        "video_max_frames": self.video_max_frames,
        "video_min_frames": self.video_min_frames,
        "base_interval": self.base_interval,
        "temporal_patch_size": self.temporal_patch_size,
    }
  