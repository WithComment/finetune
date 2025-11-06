from hydra.core.config_store import ConfigStore

from src.data.science_qa import ScienceQAConfig
from src.data.openbiomedvid import OBVConfig
from src.data.data_builder import DataConfig, DataBuilder

def register_data_configs():
  cs = ConfigStore.instance()
  cs.store(group="data", name="base_data", node=DataConfig)
  cs.store(group="data", name="base_scienceqa", node=ScienceQAConfig)
  cs.store(group="data", name="base_obv", node=OBVConfig)
