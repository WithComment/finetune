import os
from pathlib import Path
import subprocess
from ..data import avail_datasets, BenchmarkDataset

default_datasets = [
  f"{k}:test:1.0" for k in avail_datasets
  if issubclass(avail_datasets[k]['ds_class'], BenchmarkDataset)
]

default_models = [
  f"Qwen/{name}" for name in os.listdir('/projects/cft_vlm/.checkpoint/Qwen') 
  if os.path.isdir(f"/projects/cft_vlm/.checkpoint/Qwen/{name}")
]

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run predictions on multiple models/datasets.")
  parser.add_argument(
    '--command', type=str, default='bash', choices=['bash', 'sbatch'],
    help="Command to run the predictions. Use 'bash' for local execution or 'sbatch' for Slurm batch jobs."
  )
  parser.add_argument(
    '--datasets', type=str, nargs='+',
    help="List of datasets to use for predictions in the format 'dataset_name:split:ratio'."
  )
  parser.add_argument(
    '--models', type=str, nargs='+',
    help="List of model names to use for predictions."
  )
  parser.add_argument(
    '--ignore_errors', action='store_true', default=False,
    help="Ignore errors during the execution of the command."
  )
  parser.add_argument(
    '--sys_prompt', type=str, default='default', choices=['default', 'none', 'custom'],
    help="Use custom system prompt for the predictions."
  )
  args = parser.parse_args()
  datasets = args.datasets or default_datasets
  datasets = [d.split(':') for d in datasets]
  sys_prompt = args.sys_prompt
  command = args.command
  models = args.models or default_models
  for dataset in datasets:
    if len(dataset) < 2:
      dataset.append('test')
    if len(dataset) < 3:
      dataset.append('1.0')
    print(f"Dataset: {dataset[0]}, Split: {dataset[1]}, Ratio: {dataset[2]}")
  for model in models:
    print(f"Model: {model}")
  for model in models:
    for dataset in datasets:
      requeue = command == 'sbatch'
      result = subprocess.run(
        [
          args.command, str(Path(__file__).parent / 'predict.sh'),
          dataset[0], model, dataset[1], dataset[2], sys_prompt, str(requeue).lower()
        ]
      )
      if result.returncode != 0 and not args.ignore_errors:
        exit(result.returncode)