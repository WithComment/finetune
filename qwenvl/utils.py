import glob
import logging
import os
from pathlib import Path
import shutil
import sys

import torch.distributed as dist
from transformers import TrainerCallback, TrainingArguments

def get_logger(name):
  logger = logging.getLogger(name)
  logger.setLevel(logging.INFO)
  logger.propagate = False
  # Create console handler and formatter
  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  console_handler.setFormatter(formatter)
  
  if dist.is_initialized() and dist.get_rank() != 0:
    # If not rank 0, set the logger to only log warnings and errors
    logger.setLevel(logging.WARNING)
    console_handler.setLevel(logging.WARNING)

  # Add handler to logger (avoid duplicate handlers)
  if not logger.handlers:
    logger.addHandler(console_handler)
    
  
  return logger


default_logger = get_logger(__name__)

def rank0_print(msg, logger=default_logger, lvl="INFO"):
  lvl = getattr(logging, lvl.upper(), logging.INFO)
  if not dist.is_initialized() or dist.get_rank() == 0:
    logger.log(level=lvl, msg=msg)


class PruneOldStateCallback(TrainerCallback):
  """
  A custom callback to delete optimizer, scheduler, and trainer state
  from older checkpoints, keeping only the model weights.
  Called after saving.
  """

  def on_save(self, args: TrainingArguments, state, control, **kwargs):
    if not state.is_world_process_zero:
      return
    rank0_print("Running custom callback to prune old trainer states...")

    # 1. Get all checkpoint folders
    output_dir = args.output_dir
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1])
    )

    # 2. Identify all but the most recent checkpoint
    if len(checkpoints) > 1:
      checkpoints_to_prune = checkpoints[:-1]
      rank0_print(
          f"Found {len(checkpoints_to_prune)} old checkpoints to prune.")

      # Files needed for from_pretrained() - keep these
      files_to_keep = {
          "config.json",
          "generation_config.json", 
          "preprocessor_config.json",
          "video_preprocessor_config.json",
          "tokenizer.json",
          "tokenizer_config.json",
          "vocab.json",
          "merges.txt",
          "special_tokens_map.json",
          "added_tokens.json",
          "chat_template.jinja",
          "model.safetensors.index.json"
      }
      
      for checkpoint_dir in checkpoints_to_prune:
        checkpoint_path = Path(checkpoint_dir)
        rank0_print(f"Pruning checkpoint: {checkpoint_path}")
        
        for file_path in checkpoint_path.iterdir():
          if file_path.name in files_to_keep:
            continue
          if file_path.name.startswith("model-") and file_path.name.endswith(".safetensors"):
            continue
          
          if file_path.is_dir():
            rank0_print(f"  - Deleting directory: {file_path}")
            shutil.rmtree(file_path)
          else:
            rank0_print(f"  - Deleting file: {file_path}")
            file_path.unlink()
            
      rank0_print("Checkpoint pruning completed.")
    else:
      rank0_print("No old checkpoints to prune.")


def print_trainable_parameters_visual(model) -> None:
  """
  Prints the trainable status of all vision components including attention blocks and merger module.
  Outputs the indices of trainable/non-trainable blocks and the merger module status.
  """
  trainable_blocks = []
  non_trainable_blocks = []

  # Check trainable status of vision attention blocks
  for block_idx, block in enumerate(model.blocks):
    is_trainable = all(param.requires_grad for param in block.parameters())
    if is_trainable:
      trainable_blocks.append(block_idx)
    else:
      non_trainable_blocks.append(block_idx)

  # Check trainable status of merger module
  is_merger_trainable = any(
      param.requires_grad for param in model.merger.parameters())

  # Print results
  rank0_print("Vision Module - Attention Blocks:")
  rank0_print(
      f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}"
  )
  rank0_print(
      f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}"
  )
  rank0_print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(model) -> None:
  """
  Prints the trainable status of all LLM components including embeddings, layers, and normalization.
  Outputs the indices of trainable/non-trainable layers and other module statuses.
  """
  # Check embed_tokens
  model = model.language_model
  is_embed_trainable = any(
      param.requires_grad for param in model.embed_tokens.parameters()
  )
  rank0_print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

  # Check each decoder layer
  trainable_layers = []
  non_trainable_layers = []

  for layer_idx, layer in enumerate(model.layers):
    is_trainable = any(param.requires_grad for param in layer.parameters())
    if is_trainable:
      trainable_layers.append(layer_idx)
    else:
      non_trainable_layers.append(layer_idx)

  # Print layer status
  rank0_print(
      f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}"
  )
  rank0_print(
      f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}"
  )
