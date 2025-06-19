from ..utils import get_logger

logger = get_logger(__name__)

from .base import BaseDataset
from .sft import SFTDataset
from .openbiomedvid import OpenbiomedvidDataset
from .openpmc import OpenpmcDataset
from .openpmc_tiny import OpenpmcTinyDataset
from .openpmc_small import OpenpmcSmallDataset

from .benchmark import BenchmarkDataset
from .fashion_mnist import FashionMnistDataset
from .vqa import VQADataset
from .surgeryvid import SurgeryVidDataset

avail_datasets = {
  "fashion-mnist": {
    "ds_dir": "/projects/cft_vlm/datasets/fashion_mnist/data/dataset",
    "media_dir": None,
    "ds_class": FashionMnistDataset,
    "ds_key": "zalando-datasets/fashion_mnist"
  },
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
  "surgery-vid": {
    "ds_dir": "/projects/cft_vlm/datasets/surgeryvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/surgeryvid/data/vid_processed",
    "ds_class": SurgeryVidDataset,
    "ds_key": "connectthapa84/SurgeryVideoQA"
  },
  "open-pmc": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc/data/dataset",
    "media_dir": None,
    "ds_class": OpenpmcDataset,
    "ds_key": "vector-institute/open-pmc"
  },
  "open-pmc-small": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc_small/data/dataset",
    "media_dir": None,
    "ds_class": OpenpmcSmallDataset,
    "ds_key": "vector-institute/open-pmc"
  },
  "open-pmc-tiny": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc_tiny/data/dataset",
    "media_dir": None,
    "ds_class": OpenpmcTinyDataset,
    "ds_key": "vector-institute/open-pmc"
  },
  "openbiomedvid": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_segments",
    "ds_class": OpenbiomedvidDataset,
    "ds_key": "connectthapa84/OpenBiomedVid"
  }
}
