from qwenvl.data.conversation import MCCM, MNISTCM, VQACM, CaptionCM, ChexpertCM, OBVCM, TextConversationMaker
from ..utils import get_logger


logger = get_logger(__name__)
DATA_ROOT = "/scratch/xiaowenz/datasets"


avail_datasets = {
  "merged_fashion": {
    "ds_dir": f"{DATA_ROOT}/merged_fashion/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/merged_fashion",
    "cm": VQACM,
  },
  "fashion_mnist": {
    "ds_dir": f"{DATA_ROOT}/fasion_mnist/data/dataset",
    "media_dir": None,
    "ds_key": "zalando-datasets/fashion_mnist",
    "cm": MNISTCM,
  },
  "path_vqa": {
    "ds_dir": f"{DATA_ROOT}/path_vqa/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/path-vqa",
    "cm": VQACM,
  },
  "vqa_rad": {
    "ds_dir": f"{DATA_ROOT}/vqa_rad/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/vqa-rad",
    "cm": VQACM,
  },
  "slake": {
    "ds_dir": f"{DATA_ROOT}/slake/data/dataset",
    "media_dir": None,
    "ds_key": "mdwiratathya/SLAKE-vqa-english"
  },
  "openbiomedvid": {
    "ds_dir": f"{DATA_ROOT}/openbiomedvid/data/dataset",
    "media_dir": f"{DATA_ROOT}/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid",
    "cm": OBVCM,
    "qa_list_field": "qa_pairs",
  },
  "openbiomedvid_qa": {
    "ds_dir": f"{DATA_ROOT}/openbiomedvid_qa/data/dataset",
    "media_dir": f"{DATA_ROOT}/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid_qa",
    "cm": OBVCM,
    "qa_list_field": "qa_pairs",
  },
  "openbiomedvid_cap": {
    "ds_dir": f"{DATA_ROOT}/openbiomedvid_cap/data/dataset",
    "media_dir": f"{DATA_ROOT}/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid_cap",
    "cm": CaptionCM,
  },
  "surgeryvid": {
    "ds_dir": f"{DATA_ROOT}/surgeryvid/data/dataset",
    "media_dir": f"{DATA_ROOT}/surgeryvid/data/vid_processed",
    "ds_key": "withcomment/surgeryvid",
    "cm": VQACM,
  },
  "surgeryvid_small": {
    "ds_dir": f"{DATA_ROOT}/surgeryvid_small/data/dataset",
    "media_dir": f"{DATA_ROOT}/surgeryvid/data/vid_processed",
    "ds_key": "withcomment/surgeryvid_small",
    "cm": VQACM,
  },
  "chexpert": {
    "ds_dir": f"{DATA_ROOT}/chexpert/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/chexpert",
    "cm": ChexpertCM,
  },
  "chexpert_merged": {
    "ds_dir": f"{DATA_ROOT}/chexpert_merged/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/chexpert-merged",
    "cm": ChexpertCM,
  },
  "chexpert_qa_test": {
    "ds_dir": f"{DATA_ROOT}/chexpert_qa_test/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/chexpert-qa-test",
    "cm": VQACM,
  },
  "chexpert_report": {
    "ds_dir": f"{DATA_ROOT}/chexpert_report/data/dataset",
    "media_dir": None,
    "ds_key": "ayyuce/chexpert-subset",
    "cm": CaptionCM,
  },
  "medqa_cft": {
    "ds_dir": f"{DATA_ROOT}/medqa_cft/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/medqa_cft",
    "cm": TextConversationMaker,
  },
  "medqa_mc": {
    "ds_dir": f"{DATA_ROOT}/medqa_mc/data/dataset",
    "media_dir": None,
    "ds_key": "withcomment/medqa_mc",
    "cm": MCCM,
  },
}
