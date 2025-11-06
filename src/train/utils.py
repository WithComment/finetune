from pathlib import Path
from accelerate import Accelerator
import logging

import torch

from src.train.configs import CustomSFTConfig


def get_logger(name: str, accelerator: Accelerator = None) -> logging.Logger:
  """
  Get a configured logger for GRPO training scripts.
  
  Args:
    name: Logger name (typically __name__)
    accelerator: Optional Accelerator instance for main process logging
    
  Returns:
    Configured logger instance
  """
  logger = logging.getLogger(name)
  
  # Configure logging format if not already configured
  if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
  
  # If accelerator provided, add main process wrapper methods
  if accelerator is not None:
    # Store original methods
    _info = logger.info
    _warning = logger.warning
    _error = logger.error
    _critical = logger.critical
    
    # Wrap methods to only log on main process
    @accelerator.on_main_process
    def main_info(msg, *args, **kwargs):
      _info(msg, *args, **kwargs)
    
    @accelerator.on_main_process
    def main_warning(msg, *args, **kwargs):
      _warning(msg, *args, **kwargs)
    
    @accelerator.on_main_process
    def main_error(msg, *args, **kwargs):
      _error(msg, *args, **kwargs)
    
    @accelerator.on_main_process
    def main_critical(msg, *args, **kwargs):
      _critical(msg, *args, **kwargs)
    
    # Add main process methods as attributes
    logger.main_info = main_info
    logger.main_warning = main_warning
    logger.main_error = main_error
    logger.main_critical = main_critical
  else:
    logger.main_info = logger.info
    logger.main_warning = logger.warning
    logger.main_error = logger.error
    logger.main_critical = logger.critical
  
  return logger


def is_prompt_completion(item: dict) -> bool:
  return 'prompt' in item and 'completion' in item


def calc_grad_acc_steps(
    cfg: CustomSFTConfig,
) -> int:
  """Calculate gradient accumulation steps based on batch size configuration."""
  effective_batch_size = cfg.effective_batch_size
  per_device_train_batch_size = cfg.per_device_train_batch_size
  gradient_accumulation_steps = max((
      effective_batch_size //
      (per_device_train_batch_size * torch.cuda.device_count())
  ), 1)
  return gradient_accumulation_steps


def get_last_part_of_path(path: str) -> str:
  """Get the last part of a file path."""
  return Path(path).name
