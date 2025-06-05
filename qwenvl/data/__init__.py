import re
from pathlib import Path

IMG_PROMPTS = (
  "Generate a concise and descriptive caption for the provided image.",
  "Describe this image with a short, informative textual caption.",
  "Write a brief, accurate caption for the visual content shown.",
  "Create a suitable caption to accompany this specific image input.",
  "Provide a short textual summary caption reflecting this image's content.",
  "Please generate an appropriate and concise caption for this picture.",
  "Summarize the key visual elements of this image in a caption.",
  "Compose a caption that effectively describes the scene in the image.",
  "Offer a succinct caption detailing the main focus of this visual.",
  "Formulate a fitting and descriptive caption for the image presented.",
)


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
