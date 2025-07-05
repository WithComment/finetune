from ..utils import get_logger

logger = get_logger(__name__)

SYS_PROMPTS = {
  'default': "You are a helpful assistant.",
}

CFT_PROMPTS = {
}

avail_datasets = {
  "path_vqa": {
    "ds_dir": "/projects/cft_vlm/datasets/path_vqa/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/path-vqa"
  },
  "vqa_rad": {
    "ds_dir": "/projects/cft_vlm/datasets/vqa_rad/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/vqa-rad"
  },
  "slake": {
    "ds_dir": "/projects/cft_vlm/datasets/slake/data/dataset",
    "media_dir": None,
    "ds_key": "mdwiratathya/SLAKE-vqa-english"
  },
  "surgeryvid": {
    "ds_dir": "/projects/cft_vlm/datasets/surgeryvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/surgeryvid/data/vid_processed",
    "ds_key": "withcomment/surgeryvid"
  },
  "open_pmc": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc/data/dataset",
    "media_dir": None,
    "ds_key": "vector-institute/open-pmc"
  },
  "open_pmc_small": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc_small/data/dataset",
    "media_dir": None,
    "ds_key": "vector-institute/open-pmc"
  },
  "open_pmc_tiny": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc_tiny/data/dataset",
    "media_dir": None,
    "ds_key": "vector-institute/open-pmc"
  },
  "openbiomedvid": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid"
  }
}
