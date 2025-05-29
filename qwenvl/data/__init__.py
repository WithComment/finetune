import re


def parse_sampling_rate(dataset_name):
  match = re.search(r"%(\d+)$", dataset_name)
  if match:
    return int(match.group(1)) / 100.0
  return 1.0


def data_list(dataset_names):
  with open('datasets.json', 'r') as f:
    import json
    data_dict = json.load(f)
    
  config_list = []
  for dataset_name in dataset_names:
    sampling_rate = parse_sampling_rate(dataset_name)
    dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
    if dataset_name in data_dict.keys():
      config = data_dict[dataset_name].copy()
      config["sampling_rate"] = sampling_rate
      config_list.append(config)
    else:
      raise ValueError(f"Cannot find {dataset_name}")
  return config_list


if __name__ == "__main__":
  dataset_names = ["openbiomedvid"]
  configs = data_list(dataset_names)
  for config in configs:
    print(config)
