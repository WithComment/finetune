from ..utils import get_logger

logger = get_logger(__name__)

from .base import BaseDataset
from .sft import SFTDataset
from .openbiomedvid import OpenbiomedvidDataset
from .openpmc import OpenpmcDataset

from .benchmark import BenchmarkDataset
from .vqa import VQADataset

avail_datasets = {
  "path-vqa": {
    "ds_dir": "/projects/cft_vlm/datasets/path_vqa/data/dataset",
    "media_dir": None,
    "ds_class": VQADataset,
    "ds_key": "flaviagiammarino/path-vqa"
  },
  "vqa-rad": {
    "ds_dir": "/projects/cft_vlm/datasets/vqa_rad/data/dataset",
    "media_dir": None,
    "ds_class": VQADataset,
    "ds_key": "flaviagiammarino/vqa-rad"
  },
  "open-pmc": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc/data/dataset",
    "media_dir": None,
    "ds_class": OpenpmcDataset,
    "ds_key": "vector-institute/open-pmc"
  },
  "openbiomedvid": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_reencoded",
    "ds_class": OpenbiomedvidDataset,
    "ds_key": "connectthapa84/OpenBiomedVid"
  }
}
