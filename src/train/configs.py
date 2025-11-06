

import json
from dataclasses import dataclass, asdict
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from trl import SFTConfig

from src.data import DataConfig


@dataclass
class CustomSFTConfig:
  effective_batch_size: int = MISSING
  per_device_train_batch_size: int = MISSING
  max_length: int = MISSING
  packing: bool = MISSING
  completion_loss_only: bool = MISSING
  num_train_epochs: float | None = MISSING
  max_steps: int | None = MISSING

  # Optimization
  lr: float = MISSING
  lr_schedular_type: str = MISSING
  warmup_ratio: float = MISSING
  weight_decay: float = MISSING
  optim: str = MISSING

  # Memory optimization
  dataloader_num_workers: int = MISSING
  dataset_num_proc: int | None = MISSING
  group_by_length: bool = MISSING
  gradient_checkpointing: bool = MISSING
  bf16: bool = MISSING
  deepspeed: str | None = MISSING

  # Saving and logging
  save_strategy: str = MISSING
  save_steps: float = MISSING
  save_total_limit: int = MISSING
  report_to: str = MISSING
  logging_steps: int = MISSING
  run_name: str = MISSING
  output_dir: str = MISSING


def custom_to_trl_config(custom_config: CustomSFTConfig, gradient_accumulation_steps: int) -> SFTConfig:
  accepted_kwargs = set(asdict(SFTConfig()).keys())
  d = OmegaConf.to_container(custom_config, resolve=True)
  if d['save_steps'] >= 1:
    d['save_steps'] = int(d['save_steps'])
  kwargs = {
      k: v for k, v in d.items()
      if k in accepted_kwargs
  }
  if kwargs.get('deepspeed'):
    with open(kwargs['deepspeed'], 'r') as f:
      kwargs['deepspeed'] = json.load(f)
  return SFTConfig(
      **kwargs,
      gradient_accumulation_steps=gradient_accumulation_steps
  )


@dataclass
class Config:
  base_run_name: str = MISSING
  base_run_dir: str = MISSING
  run_dir: str = MISSING
  model_path: str = MISSING
  trainer: CustomSFTConfig = MISSING
  data: DataConfig = MISSING


def register_configs():
  cs = ConfigStore.instance()
  cs.store(group="trainer", name="base_trainer", node=CustomSFTConfig)
  cs.store(name="base_config", node=Config)
