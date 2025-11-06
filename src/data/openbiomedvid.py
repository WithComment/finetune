from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable, Optional

from datasets import Dataset, interleave_datasets, concatenate_datasets, IterableDataset
from omegaconf import MISSING


def is_video_corrupted(video_path: str) -> bool:
  cmd = ['ffmpeg', '-v', 'error', '-i', video_path, '-f', 'null', '-xerror', '-']
  result = subprocess.run(cmd)
  return result.returncode != 0


@dataclass
class OBVConfig:
  data_dir: str
  category_filters: Optional[dict[str, list[str]]] = None
  prob_with_context: float = MISSING

class OBVBuilder:
  
  data_dir: Path
  
  def __init__(self, **kwargs):
    '''
    Do not call directly. Use `hydra.utils.instantiate`.
    '''
    self.__dict__.update(kwargs)
    self.data_dir = Path(self.data_dir)
    

  def video_path(self, item: dict) -> str:
    return str(self.data_dir / item['video'])

  
  def filter(self, item: dict) -> bool:
    return (
      not is_video_corrupted(self.video_path(item))
      and (self.modality is None or item['modality'].lower() in self.modality)
      and (self.anatomical_region is None or item['anatomical_region'].lower() in self.anatomical_region)
    )


  def proc_ds(self, ds: Dataset) -> Dataset:
    return ds
    
  def map_qa(self, item: dict) -> dict:
    prompt = [
        {"role": "user", "content": [
          {"type": "video", "video": self.video_path(item)},
        ]}
    ]
    
    completion = [
        {"role": "assistant", "content": [
          {"type": "text", "text": '\n\n'.join([f"Question: {pair['question']}\nAnswer: {pair['choices']}" for pair in item['qa_pairs']])}
        ]}
    ]
    return {"prompt": prompt, "completion": completion}
  
  def map_caption(self, item: dict) -> dict:
    prompt = [{
        "role": "user", 
        "content": [{"type": "video", "video": self.video_path(item)}]
    }]
    
    completion = [{
        "role": "assistant", 
        "content": [{"type": "text", "text": item['caption']}]
    }]
    return {"prompt": prompt, "completion": completion}
  
  def __call__(self, ds: Dataset | None = None, is_main_process: bool = True) -> Dataset:
    if self.streaming or isinstance(ds, IterableDataset):
      raise NotImplementedError("Streaming not supported for OpenBioMedVid yet.")
    
    if ds is None:
      # TODO add support for only qa/caption
      ds = super().load()
    ds = ds.filter(self.filter, load_from_cache_file=(not is_main_process), num_proc=self.num_proc)
    ds_qa = ds.map(self.map_qa, load_from_cache_file=(not is_main_process), num_proc=self.num_proc)
    ds_cap = ds.map(self.map_caption, load_from_cache_file=(not is_main_process), num_proc=self.num_proc)
    ds = interleave_datasets([ds_qa, ds_cap])
    return ds
  