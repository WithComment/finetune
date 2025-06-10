import json
from pathlib import Path
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer
from qwenvl.new.argument import *
from qwenvl.new.data import avail_datasets, BenchmarkDataset

from .utils import get_logger
from .train import rank0_make_data_module, set_processor

import logging
import torch.distributed as dist

logger = get_logger(__name__) 

def rank0_print(*args, lvl="INFO"):
  lvl = getattr(logging, lvl.upper(), logging.INFO)
  if not dist.is_initialized() or dist.get_rank() == 0:
    logger.log(level=lvl, *args)


def get_trainer_args(eval_args):
  return Seq2SeqTrainingArguments(
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    do_train=False,
    do_eval=False,
    do_predict=True,
    generation_config=GenerationConfig(
      max_new_tokens=eval_args.max_new_tokens,
      do_sample=eval_args.do_sample,
    )
  )


def get_generated_text(model_output, tokenizer):
  np_array = model_output.predictions.copy()
  np_array[np_array == -100] = tokenizer.pad_token_id
  texts = tokenizer.batch_decode(np_array, skip_special_tokens=True)
  generated = list()
  for text in texts:
    text = text.split("assistant")[-1].strip().strip("\n")
    generated.append(text)
  return generated


def save_result(
    dataset: BenchmarkDataset,
    generated_text: list[str],
    output_dir: Path
) -> Path:
  output_dir.mkdir(parents=True, exist_ok=True)
  output_file_path = output_dir / "eval_results.json"
  results = []
  for item, text in zip(dataset, generated_text):
    item['model_output'] = text
    results.append(dataset.drop_non_json_fields(item))
  with open(output_file_path, 'w') as f:
    json.dump(results, f, indent=2)
    
  return output_file_path


def predict(data_args, proc_args, eval_args):
  
  model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
      eval_args.model_name_or_path,
      torch_dtype='auto',
      attn_implementation="flash_attention_2"
  )
  processor = AutoProcessor.from_pretrained(eval_args.model_name_or_path)
  processor = set_processor(processor, proc_args, data_args)
  data_module = rank0_make_data_module(
      processor=processor,
      data_args=data_args,
      proc_args=proc_args,
      for_training=False
  )
  trainer_args = get_trainer_args(eval_args)

  trainer = Seq2SeqTrainer(
      model=model,
      processing_class=processor,
      args=trainer_args,
      **data_module
  )
  output = trainer.predict(trainer.eval_dataset)
  generated_text = get_generated_text(output, processor.tokenizer)
  if eval_args.output_dir is not None:
    output_dir = Path(eval_args.output_dir)
  else:
    output_dir = Path(eval_args.model_name_or_path)
  save_result(data_module['eval_dataset'], generated_text, output_dir)
      
if __name__ == "__main__":
  parser = transformers.HfArgumentParser((DataArguments, ProcessingArguments, EvalArguments))
  data_args, proc_args, eval_args = parser.parse_args_into_dataclasses()
  predict(data_args, proc_args, eval_args)