import random

from .openpmc import OpenpmcDataset

from ..utils import get_logger

logger = get_logger(__name__)

class OpenpmcSmallDataset(OpenpmcDataset):
  @staticmethod
  def _preprocess(ds, media_dir, num_proc):
    logger.info("Preprocessing OpenPMC small dataset")
    for k in ds:
      n = len(ds[k])
      ds[k] = ds[k].select(random.sample(range(n), int(n * 0.1)))
    
    ds = OpenpmcDataset._preprocess(ds, media_dir, num_proc)
    return ds
