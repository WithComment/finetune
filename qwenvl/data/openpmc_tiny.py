import random

from .openpmc import OpenpmcDataset

class OpenpmcTinyDataset(OpenpmcDataset):
  @staticmethod
  def _preprocess(ds, media_dir, num_proc):
    for k in ds:
      n = len(ds[k])
      ds[k] = ds[k].select(random.sample(range(n), int(n * 0.01)))
    ds = OpenpmcDataset._preprocess(ds, media_dir, num_proc)
    return ds
