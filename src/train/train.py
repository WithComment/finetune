import os
from pathlib import Path
from pprint import pformat

from accelerate import Accelerator
from accelerate.utils import broadcast_object_list
from omegaconf import OmegaConf
import hydra
from hydra.utils import instantiate
from trl import SFTTrainer
from trl.data_utils import is_conversational

from src.data import register_data_configs, DataBuilder
from src.train.configs import custom_to_trl_config, register_configs, Config
from src.train.utils import calc_grad_acc_steps, get_last_part_of_path, get_logger

os.environ["WANDB_PROJECT"] = "finetune"
accelerator = Accelerator()
logger = get_logger(__name__, accelerator)
register_configs()
register_data_configs()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: Config):
  logger.main_info("Launching GRPO pipeline with configuration:")
  logger.main_info(OmegaConf.to_yaml(cfg, resolve=True))
  logger.main_info("Instatiating dataset...")
  logger.main_info("Building data...")
  data_builder: DataBuilder = instantiate(cfg.data)
  with accelerator.main_process_first():
    ds = data_builder(is_main_process=accelerator.is_main_process)
  logger.main_info(f"Preprocessed dataset:\n{pformat(ds)}")
  logger.main_info(f"Features:\n{pformat(ds.features)}")
  example = next(iter(ds))
  logger.main_info(f"Sample after preprocessing:\n{pformat(example):.1000}\n{pformat(ds[3])}")
  logger.main_info("Is conversational: " + str(is_conversational(example)))
  grad_acc_steps = calc_grad_acc_steps(cfg.trainer)
  trainer_config = custom_to_trl_config(cfg.trainer, grad_acc_steps)

  trainer = SFTTrainer(
      model=cfg.model_path,
      train_dataset=ds,
      args=trainer_config,
  )
  
  trainer.train()
  accelerator.wait_for_everyone()
  trainer.save_model()
  logger.main_info(" Training completed ".center(60, '='))


if __name__ == "__main__":
  main()
