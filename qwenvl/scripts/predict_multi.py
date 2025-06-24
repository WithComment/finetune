import argparse
import subprocess
import sys

# Default values
DEFAULT_DATASET_SPLITS = [
    ("surgery-vid", "test", 1.0),
    ("path_vqa", "test", 1.0),
    ("vqa_rad", "train", 1.0),
]
DEFAULT_MODEL_NAMES = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
]


def parse_args():
  parser = argparse.ArgumentParser(
      description="Schedule multiple predict.sh jobs with sbatch.")
  parser.add_argument(
      "--dataset_splits",
      type=str,
      help="Comma-separated list of dataset:split:portion triples, e.g. dataset1:val:0.5,dataset2:test:1.0"
  )
  parser.add_argument(
      "--model_names",
      type=str,
      help="Comma-separated list of model names"
  )
  return parser.parse_args()


def main():
  args = parse_args()

  if args.dataset_splits:
    try:
      dataset_splits = []
      for pair in args.dataset_splits.split(","):
        parts = pair.split(":")
        if len(parts) == 3:
          dataset, split, portion = parts
          portion = float(portion)
        elif len(parts) == 2:
          dataset, split = parts
          portion = 1.0
        else:
          raise ValueError
        dataset_splits.append((dataset, split, portion))
    except Exception:
      print(
          "Error: --dataset_splits must be in the format dataset:split[:portion], e.g. dataset1:val:0.5,dataset2:test")
      sys.exit(1)
  else:
    dataset_splits = DEFAULT_DATASET_SPLITS

  if args.model_names:
    model_names = args.model_names.split(",")
  else:
    model_names = DEFAULT_MODEL_NAMES

  script_path = "predict.sh"

  for dataset_name, split, portion in dataset_splits:
    for model_name in model_names:
      cmd = [
          "sbatch",
          script_path,
          dataset_name,
          split,
          model_name,
          str(portion)
      ]
      print(f"Scheduling: {' '.join(cmd)}")
      subprocess.run(cmd)


if __name__ == "__main__":
  main()
