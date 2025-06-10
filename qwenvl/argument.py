import transformers
from dataclasses import dataclass, field
from typing import Optional

PATCH_WIDTH = 28
PATCH_SIZE = PATCH_WIDTH ** 2


@dataclass
class ModelArguments:
  model_name_or_path: Optional[str] = field(
      default="Qwen/Qwen2.5-VL-3B-Instruct")
  tune_mm_llm: bool = field(default=False)
  tune_mm_mlp: bool = field(default=False)
  tune_mm_vision: bool = field(default=False)


@dataclass
class DataArguments:
  dataset_use: str = field(default="")
  data_packing: bool = field(default=True)
  split: str = field(default="train")
  use_cft: bool = field(default=False)
  model_max_length: int = field(default=3072)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
  cache_dir: Optional[str] = field(default=None)
  optim: str = field(default="adamw_torch")
  mm_projector_lr: Optional[float] = None
  vision_tower_lr: Optional[float] = None


@dataclass
class ProcessingArguments:
  """
  These arguments are important for content token count.
  They will be saved.
  """
  image_min_pixels: int = field(default=PATCH_SIZE * 8 * 8)
  image_max_pixels: int = field(default=PATCH_SIZE * 8 * 40)
  shortest_edge: int = field(default=PATCH_WIDTH * 8)

  video_min_pixels: int = field(default=PATCH_SIZE * 8 * 8)
  video_max_pixels: int = field(default=PATCH_SIZE * 8 * 24)
  video_default_to_square: bool = field(default=False)
  video_max_frames: int = field(default=8)
  video_min_frames: int = field(default=4)
  base_interval: int = field(default=2)


@dataclass
class EvalArguments:
  model_name_or_path: str = field(
      default="Qwen/Qwen2.5-VL-3B-Instruct",
      metadata={"help": "Path to the model to be evaluated."}
  )
  max_new_tokens: int = field(
      default=32,
      metadata={"help": "Maximum number of new tokens to generate."}
  )
  do_sample: bool = field(
      default=False,
      metadata={"help": "Whether to use sampling for generation."}
  )
  output_dir: str = field(
      default=None,
      metadata={"help": "Directory to save the evaluation results."}
  )
  # deepspeed: str = field(
  #     default=None,
  #     metadata={"help": "Path to the deepspeed configuration file."}
  # )