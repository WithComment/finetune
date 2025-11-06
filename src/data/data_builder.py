from dataclasses import dataclass
import io
import os
from typing import Optional
from datasets import Dataset, IterableDataset, load_dataset, Features
from omegaconf import MISSING
from PIL import Image as PILImage


@dataclass
class DataConfig:
  _target_: str = MISSING
  path: str = MISSING
  name: Optional[str] = MISSING
  split: str = MISSING
  streaming: bool = MISSING
  load_from_cache_file: bool = MISSING
  num_proc: int = MISSING
  input_format: list[str] | None = MISSING
  
  
def is_dict_of_lists(d: dict) -> bool:
  return all(isinstance(v, list) for v in d.values())


def is_list_of_dicts(l: list) -> bool:
  return all(isinstance(i, dict) for i in l)


def dl_to_ld(dl: dict) -> list[dict]:
  length = len(next(iter(dl.values())))
  return [{k: dl[k][i] for k in dl} for i in range(length)]


def ld_to_dl(ld: list[dict]) -> dict:
  if not ld:
    return {}
  return {k: [d[k] for d in ld] for k in ld[0]}


class DataBuilder:

  path: str
  name: str | None
  split: str
  streaming: bool
  featuers: Features | None = None

  def __init__(self, **kwargs):
    '''
    Do not call directly. Use `hydra.utils.instantiate`.
    '''
    self.__dict__.update(kwargs)
    self.features = None

  def load(self) -> Dataset | IterableDataset:
    return load_dataset(
        path=self.path,
        name=self.name,
        split=self.split,
        streaming=self.streaming
    )

  def filter(self, item: dict) -> bool:
    raise NotImplementedError
    
  def _to_pil(self, img) -> PILImage.Image | None:
    if img is None:
      return None
    if isinstance(img, PILImage.Image):
      return img
    if isinstance(img, dict):
      b = img.get("bytes")
      p = img.get("path")
      if b is not None:
        return PILImage.open(io.BytesIO(b)).convert("RGB")
      if p:
        return PILImage.open(p).convert("RGB")
      return None
    if isinstance(img, (bytes, bytearray)):
      return PILImage.open(io.BytesIO(img)).convert("RGB")
    if isinstance(img, str) and os.path.exists(img):
      return PILImage.open(img).convert("RGB")
    return img


  def map(self, item: dict[str, list]) -> dict[str, list]:
    if is_dict_of_lists(item):
      ld = dl_to_ld(item)
      mapped_ld = [self._map(d) for d in ld]
      return ld_to_dl(mapped_ld)
    return self._map(item)
  
  
  def _transform(self, item: dict[str]) -> dict[str]:
    if 'prompt' in item:
      new_prompt = []
      for message in item['prompt']:
        new_message = message.copy()
        new_contents = []
        for content in message['content']:
          if 'image' in content:
            content['image'] = self._to_pil(content['image'])
          for k in list(content.keys()):
            if content[k] is None:
              content.pop(k)
          new_contents.append(content)
        new_message['content'] = new_contents
        new_prompt.append(new_message)
      item['prompt'] = new_prompt
    return item
  
  
  def transform(self, item: dict[str, list]) -> dict[str, list]:
    if is_dict_of_lists(item):
      ld = dl_to_ld(item)
      mapped_ld = [self._transform(d) for d in ld]
      return ld_to_dl(mapped_ld)
    return self._transform(item)
  

  def __call__(self, ds: Dataset | IterableDataset | None = None, is_main_process: bool = True) -> Dataset | IterableDataset:
    if ds is None:
      ds = self.load()
    if self.streaming:
      ds = ds.filter(self.filter)
      ds = ds.map(self.map)
    else:
      ds = ds.filter(self.filter, num_proc=self.num_proc, load_from_cache_file=not is_main_process or self.load_from_cache_file)
      ds = ds.map(self.map, num_proc=self.num_proc, load_from_cache_file=not is_main_process or self.load_from_cache_file, remove_columns=ds.column_names, features=self.features)
    
    return ds.with_transform(self.transform)
  