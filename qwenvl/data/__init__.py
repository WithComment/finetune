import re
from pathlib import Path

from qwenvl.data.benchmark import Benchmark
from qwenvl.data.openbiomedvid import OpenbiomedvidDataset
from qwenvl.data.openpmc import OpenpmcDataset
from qwenvl.data.sft_dataset import SFTDataset
from qwenvl.data.vqa import VQADataset

dataset_classes: dict[str, SFTDataset] = {
    'openpmc': OpenpmcDataset,
    'openbiomedvid': OpenbiomedvidDataset,
}
benchmark_classes: dict[str, Benchmark] = {
    'vqa-rad': VQADataset,
    'path-vqa': VQADataset,
}

def parse_sampling_rate(dataset_name):
  match = re.search(r"%(\d+)$", dataset_name)
  if match:
    return int(match.group(1)) / 100.0
  return 1.0


def data_list(dataset_names):
  ds_path = Path(__file__).parent / 'datasets.json'
  with open(ds_path, 'r') as f:
    import json
    data_dict = json.load(f)
    
  config_list = []
  for ds_name in dataset_names:
    if '%' in ds_name:
      ds_name, sampling_rate = ds_name.split('%')
      sampling_rate = float(sampling_rate) / 100
    else:
      sampling_rate = 1.0
    
    if ds_name in data_dict.keys():
      config = data_dict[ds_name].copy()
      config["sampling_rate"] = sampling_rate
      config_list.append(config)
    else:
      raise ValueError(f"Cannot find {ds_name}")
  return config_list


if __name__ == "__main__":
  dataset_names = ["openbiomedvid"]
  configs = data_list(dataset_names)
  for config in configs:
    print(config)
