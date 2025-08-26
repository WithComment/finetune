# Utility: load a model (optional) and its matching template given a model id/key.
# Uses only functions defined in this repository.
from dataclasses import dataclass
import copy
from qwenvl.data import avail_datasets
from tqdm import trange
import re
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from regex import template
import datasets
from typing import Callable, Iterator
import json
from typing import Optional, Any
import transformers
import os

from swift.llm import get_model_tokenizer, get_template, Template
from swift.llm.utils import to_device


def load_model_and_template(
  model_id_or_path: str,
  sys_prompt: Optional[str] = None,
  model_type: str = 'qwen2_5_vl',
  mode: str = 'train'
) -> tuple[transformers.PreTrainedModel, transformers.AutoProcessor, Template]:
  model, processor = get_model_tokenizer(
      model_id_or_path, model_type=model_type, load_model=True)

  if sys_prompt is not None and os.path.exists(sys_prompt):
    with open(sys_prompt, 'r') as f:
      sys_prompt = f.read()

  template = get_template(template_type=model_type,
                          processor=processor, default_system=sys_prompt)
  template.set_mode(mode)
  template.model = model
  return model, processor, template


def cleanup_dict(d: dict):
  for k in list(d.keys()):
      del d[k]
  del d
  torch.cuda.empty_cache()


def cat_grads(g: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
  if isinstance(g, torch.Tensor):
    return g.view(-1)
  return torch.cat([v.flatten().detach().cpu() for v in g.values()])


def cosine_alignment(g_train: dict[str, torch.Tensor], g_star: dict[str, torch.Tensor] | torch.Tensor, clean_g_star: bool = True, eps: float = 1e-8) -> torch.Tensor:
  g_train_cat = cat_grads(g_train)
  cleanup_dict(g_train)
  if isinstance(g_star, dict):
    g_star_cat = cat_grads(g_star)
  else:
    g_star_cat = g_star.view(-1).detach().cpu()
  sim = F.cosine_similarity(g_train_cat, g_star_cat, dim=0, eps=eps)
  del g_train_cat, g_star_cat
  torch.cuda.empty_cache()
  return sim


def ng_cosine_alignment(g_train: dict[str, torch.Tensor], g_star: dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
  raise NotImplementedError


def select_params(model, pattern) -> dict[str, Any]:
  pattern = re.compile(pattern)
  params = {}
  for name, param in model.named_parameters():
    if pattern.search(name):
      param.requires_grad = True
      params[name] = param
    else:
      param.requires_grad = False

  if not params:
    raise ValueError(f"No parameters matched pattern: {pattern}")
  return params


def load_model_collate_fn(model_path_or_key: str, sys_prompt: str | None = None, model_type: str = 'qwen2_5_vl') -> tuple[transformers.PreTrainedModel, Callable]:
  '''
  Load model, processor, template using swift. Then create a collate function
  that turns item in dataset to model input with labels.
  '''
  model, processor, template = load_model_and_template(
      model_path_or_key, sys_prompt=sys_prompt, model_type=model_type, mode='train')

  def collate_fn(rows):
    encoded = [template.encode(r) for r in rows]
    return to_device(template.data_collator(encoded), model.device)
  return model, collate_fn


def get_grad_of_batch(
    model,
    batch,
    params: dict[str, torch.Tensor],
    zero_grad: bool = True,
) -> dict[str, torch.Tensor]:
  if zero_grad:
    model.zero_grad()
  outputs = model(**batch)
  loss = outputs.loss
  loss.backward()
  return {name: param.grad for name, param in params.items()}


def accum_grad(model, batches: Iterator, n_batches: int, params: dict[str, torch.Tensor], prog_bar=False) -> dict[str, torch.Tensor]:
  model.zero_grad()
  for _ in trange(n_batches, disable=not prog_bar, desc='accum grad loop'):
    batch = next(batches)
    get_grad_of_batch(model, batch, params, zero_grad=False)
  result = {name: param.grad.detach().cpu() / n_batches for name,
            param in params.items()}
  torch.cuda.empty_cache()
  return result


@dataclass
class Experiment:
  model_path_or_key: str
  b_size: int
  train_step_size: int
  train_n_samples: int
  val_step_size: int
  val_n_samples: int
  param_pattern: str
  proxy_fn: Callable[[], torch.Tensor]
  align_fn: Callable[[], torch.Tensor]
  max_length: int = 800
  lr: float = 1e-5
  optimizer: Callable | None = None
  result: float = -torch.inf
  ds_name: str = "path_vqa"
  ds_split: str = "train"
  sys_prompt: str = "default"
  seed: int = 42

  def toJSON(self):
    return {
      'model_path_or_key': self.model_path_or_key,
      'b_size': self.b_size,
      'train_step_size': self.train_step_size,
      'train_n_samples': self.train_n_samples,
      'val_step_size': self.val_step_size,
      'val_n_samples': self.val_n_samples,
      'param_pattern': self.param_pattern,
      'proxy_fn': getattr(self.proxy_fn, '__name__', str(self.proxy_fn)),
      'align_fn': getattr(self.align_fn, '__name__', str(self.align_fn)),
      'max_length': self.max_length,
      'lr': self.lr,
      'optimizer': getattr(self.optimizer, '__name__', None) if self.optimizer is not None else None,
      'result': self.result.item() if isinstance(self.result, torch.Tensor) else self.result,
      'ds_name': self.ds_name,
      'ds_split': self.ds_split,
      'sys_prompt': self.sys_prompt,
      'seed': self.seed,
    }

  def __str__(self):
    return str(self.toJSON())


def run_experiment(
    exp: Experiment,
    g_star: dict[str, torch.Tensor] | torch.Tensor | None = None,
) -> Experiment:
  transformers.enable_full_determinism(exp.seed)

  if exp.proxy_fn is accum_grad:
    exp.result = torch.zeros(1)

  # Load model, processor, template
  # Create collate function
  model, collate_fn = load_model_collate_fn(
    exp.model_path_or_key, exp.sys_prompt)

  # Load train_ds, val_ds
  ds_key = avail_datasets[exp.ds_name]['ds_key']
  ds = datasets.load_dataset(ds_key)[exp.ds_split]
  if exp.max_length is not None:
    ds = ds.filter(lambda x: x['length'] <= exp.max_length)
  ds = ds.train_test_split(train_size=exp.train_n_samples)
  train_ds, val_ds = ds.values()
  print(train_ds, val_ds)
  # Make dataloader
  train_iter = iter(DataLoader(
    train_ds, batch_size=exp.b_size, collate_fn=collate_fn))
  val_iter = iter(DataLoader(
    val_ds, batch_size=exp.b_size, collate_fn=collate_fn))
  train_batches_per_step = max(exp.train_step_size // exp.b_size, 1)
  val_batches_per_step = max(exp.val_step_size // exp.b_size, 1)
  # Get parameters of interest. Set requires_grad to false for those not of interest.
  params = select_params(model, exp.param_pattern)
  if exp.optimizer is not None:
    optimizer = exp.optimizer(params.values(), lr=exp.lr)

  else:
    # We can calculate gradient proxy upfront.
    optimizer = None
    if g_star is None:
      g_star = exp.proxy_fn(
        model, val_iter, val_batches_per_step, params, prog_bar=True)
      g_star = cat_grads(g_star)
    model.zero_grad()

  # Main loop
  for _ in trange(exp.train_n_samples // exp.train_step_size, desc='Main loop'):

    # Get grad of one training batch
    g_train = accum_grad(model, train_iter, train_batches_per_step, params)
    if optimizer is not None:
      optimizer.step()
      model.zero_grad()

      # Get proxy for true gradient
      g_star = exp.proxy_fn(model, val_iter, val_batches_per_step, params)
    model.zero_grad()

    # Calculate alignment
    exp.result += exp.align_fn(g_train, g_star).cpu()

  # Cleanup all gpu memory
  del model
  del params
  del g_train
  torch.cuda.empty_cache()

  # Normalize by number of batches.
  exp.result /= len(train_iter)

  return exp, g_star


exp_default = Experiment(
  model_path_or_key='/scratch/xiaowenz/checkpoints/Qwen2_5_3',
  b_size=2,
  train_step_size=32,
  train_n_samples=5000,
  val_step_size=10000,
  val_n_samples=10000,
  param_pattern=r'.+language_model.+mlp.+weight$',
  proxy_fn=accum_grad,
  align_fn=cosine_alignment,
  lr=1e-5,
  optimizer=None,
  ds_name='swift_path_vqa',
  ds_split='train',
  sys_prompt='/home/xiaowenz/finetune/qwenvl/data/prompts/default.txt',
  seed=42,
  max_length=750,
)

exp_default, g_star = run_experiment(exp_default)
json.dump(exp_default.toJSON(), open('experiments/exp_default.json', 'w'), indent=2)

exp_v3_5 = copy.deepcopy(exp_default)
exp_v3_5.sys_prompt = '/home/xiaowenz/finetune/qwenvl/data/prompts/v3_5.txt'

exp_v3_5, g_star = run_experiment(exp_v3_5, g_star)
json.dump(exp_v3_5.toJSON(), open('experiments/exp_v3_5.json', 'w'), indent=2)
