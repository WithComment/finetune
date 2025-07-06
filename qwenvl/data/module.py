import itertools
import random
from typing import Callable
import datasets
from transformers import AutoProcessor
from qwenvl.argument import DataArguments, ProcessingArguments
from qwenvl.data.conversation import AllPromptAdder, ConversationMaker, ConversationProcessor, FirstPromptAdder, RolePromptAdder, VQAConversationMaker
from qwenvl.data.input_processor import InputProcessor
from qwenvl.data.preprocess import GetNumMediaTokensStrategy, GetNumTokensStrategy, PreprocessStrategy, SaveStrategy, pack_dataset
from qwenvl.data import CFT_PROMPTS, SYS_PROMPTS, avail_datasets
from qwenvl.utils import get_logger

logger = get_logger(__name__)


class DatasetWrapper:
  def __init__(self, ds: datasets.Dataset, bins: list | None):
    self.ds = ds
    self.bins = bins
  
  def __getitem__(self, idx: int) -> list[dict]:
    if isinstance(idx, slice):
      return [[self.ds[i] for i in _bin] for _bin in self.bins[idx]]
    return [self.ds[i] for i in self.bins[idx]]
  
  def __len__(self) -> int:
    return len(self.bins)


def create_module(
    data_args: DataArguments,
    preprocess_strategies: list[PreprocessStrategy],
    cp: ConversationProcessor,
    ip: InputProcessor,
    rank: int = 0
) -> tuple[DatasetWrapper, Callable[[list[dict]], dict]]:
  """
  Assembles everything, from arguments and dataset and preprocess strategies, into a module,
  as well as a collate function, that will be used directly for the trainer.
  """
  logger.info(f"Creating module for dataset {data_args.dataset_use} with split {data_args.split}")
  if rank == 0:
    ds = datasets.load_dataset(avail_datasets[data_args.dataset_use]['ds_key'])
  else:
    ds = datasets.load_from_disk(avail_datasets[data_args.dataset_use]['ds_dir'])
  
  for strategy in preprocess_strategies:
    ds = strategy(ds)

  ds = ds[data_args.split]
  
  if data_args.packing:
    bins = pack_dataset(ds, data_args.model_max_length)
  else:
    bins = [[i] for i in range(len(ds))]
  
  ds = DatasetWrapper(ds, bins)
  
  def collate_fn(batch):
    conv = [cp(item) for item in batch]
    return ip(conv)
  
  example = random.choice(ds)[:1]
  conv = cp(example)
  text = ip.get_text(conv, text_only=True)
  
  
  logger.info(f"Example from dataset: {example}")
  logger.info(f"Example after conversation processing: {conv}")
  logger.info(f"Example after input processing: {text}")
  
  logger.info(f"Finished creating data module.")
  return ds, collate_fn


def create_strategies(
    processor: AutoProcessor,
    data_args: DataArguments,
    proc_args: ProcessingArguments,
    rank: int,
) -> tuple[list[PreprocessStrategy], ConversationMaker, InputProcessor]:
  """
  Create preprocess strategies and conversation maker.
  """
  if 'vqa' not in data_args.dataset_use:
    raise NotImplementedError(
        f"Dataset {data_args.dataset_use} is not supported yet."
    )
  ds_config = avail_datasets[data_args.dataset_use]
  
  base_cm = VQAConversationMaker(for_training=data_args.split == 'train')
  modifiers = []
  
  # Sysprompt must come after cft prompts because we don't want cft prompts on system prompt.
  if proc_args.cft_prompt and proc_args.cft_prompt in CFT_PROMPTS:
    cft_prompts = CFT_PROMPTS[proc_args.cft_prompt]
    modifiers.append(AllPromptAdder(cft_prompts))
  else:
    logger.warning(f"CFT prompt {proc_args.cft_prompt} not found, not using it.")
    logger.warning(f"Available CFT prompts: {list(CFT_PROMPTS.keys())}")
  
  if data_args.split != 'train':
    modifiers.append(RolePromptAdder(["Answer straightforwardly and concisely: "], idx=1, role='user'))
    
  if proc_args.sys_prompt and proc_args.sys_prompt in SYS_PROMPTS:
    sys_prompt = SYS_PROMPTS[proc_args.sys_prompt]
    modifiers.append(FirstPromptAdder(sys_prompt))
  else:
    logger.warning(f"System prompt {proc_args.sys_prompt} not found, not using it.")
  cp = ConversationProcessor(
      conversation_maker=base_cm,
      conversation_modifiers=modifiers,
  )
  
  ip = InputProcessor(
      processor=processor,
      config=proc_args,
  )
  
  preprocess_strategies = []
  if rank == 0:
    if data_args.packing:
      preprocess_strategies.append(GetNumMediaTokensStrategy(
          get_content_fn=cp.get_content,
          config=proc_args,
      ))
      preprocess_strategies.append(GetNumTokensStrategy(cm=cp, processor=ip))
    preprocess_strategies.append(SaveStrategy(ds_config['ds_dir'], ds_config['ds_dir']))
  
  return preprocess_strategies, cp, ip