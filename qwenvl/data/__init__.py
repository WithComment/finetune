from qwenvl.data.conversation import MCCM, VQACM, CaptionCM, ChexpertCM, ClassificationCM, OBVCM, TextConversationMaker
from ..utils import get_logger

logger = get_logger(__name__)


avail_datasets = {
  "path_vqa": {
    "ds_dir": "/projects/cft_vlm/datasets/path_vqa/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/path-vqa",
    "cm": VQACM,
  },
  "vqa_rad": {
    "ds_dir": "/projects/cft_vlm/datasets/vqa_rad/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/vqa-rad",
    "cm": VQACM,
  },
  "slake": {
    "ds_dir": "/projects/cft_vlm/datasets/slake/data/dataset",
    "media_dir": None,
    "ds_key": "mdwiratathya/SLAKE-vqa-english"
  },
  "openbiomedvid": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid",
    "cm": OBVCM,
    "cm_kwargs": {
      "qa_list_field": "qa_pairs",
    }
  },
  "openbiomedvid_qa": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid_qa/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid_qa",
    "cm": OBVCM,
    "cm_kwargs": {
      "qa_list_field": "qa_pairs",
    }
  },
  "openbiomedvid_cap": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid_cap/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid_cap",
    "cm": CaptionCM,
  },
  "surgeryvid": {
    "ds_dir": "/projects/cft_vlm/datasets/surgeryvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/surgeryvid/data/vid_processed",
    "ds_key": "withcomment/surgeryvid",
    "cm": VQACM,
  },
  "chexpert": {
    "ds_dir": "/projects/cft_vlm/datasets/chexpert/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/chexpert",
    "cm": ChexpertCM,
  },
  "chexpert_merged": {
    "ds_dir": "/projects/cft_vlm/datasets/chexpert_merged/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/chexpert-merged",
    "cm": ChexpertCM,
  },
  "chexpert_qa_test": {
    "ds_dir": "/projects/cft_vlm/datasets/chexpert_qa_test/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/chexpert-qa-test",
    "cm": VQACM,
  },
  "chexpert_report": {
    "ds_dir": "/projects/cft_vlm/datasets/chexpert_report/data/dataset",
    "media_dir": None,
    "ds_key": "ayyuce/chexpert-subset",
    "cm": CaptionCM,
  },
  "medqa_cft": {
    "ds_dir": "/projects/cft_vlm/datasets/medqa_cft/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/medqa_cft",
    "cm": TextConversationMaker,
  },
  "medqa_mc": {
    "ds_dir": "/projects/cft_vlm/datasets/medqa_mc/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/medqa_mc",
    "cm": MCCM,
  },
}
