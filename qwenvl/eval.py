import json
from typing import Callable
from pathlib import Path
import re

from .utils import get_logger

logger = get_logger(__name__)


def evaluate(output_file: Path, comp_answer: Callable, filter: Callable = None) -> dict:
  items = list()
  with open(output_file, 'r') as f:
    items = [json.loads(l) for l in f]
    
  total = 0
  correct = 0
  incorrect = 0
  invalid = 0
  
  for item in items:
    if filter and not filter(item['answer']):
      continue
    total += 1
    try:
      model_answer = item.get('model_answer') or item.get('model_output', '')
      if comp_answer(item['answer'], model_answer):
        correct += 1
      else:
        incorrect += 1
    except Exception as e:
      print(f"Error processing item {item}: {e}")
      invalid += 1
  
  if total == 0:
    raise ValueError("No items found in the output file.")
  logger.info(f"Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Invalid: {invalid}")
  logger.info(f"Accuracy: {correct / total * 100:.2f}%")
  return {
    'total': total,
    'correct': correct,
    'incorrect': incorrect,
    'invalid': invalid,
    'accuracy': correct / total
  }


def yes_no_filter(answer: str) -> bool:
  return answer.strip().lower() in ['yes', 'no']


def comp_answer_basic(answer: str, model_answer: str) -> bool:
  """
  Basic comparison function that checks if the model answer matches the expected answer.
  """
  model_answer = re.split(r'[^\w]+', model_answer.strip())
  if model_answer:
    model_answer = model_answer[0]
  else:
    model_answer = ""
  return answer.strip().lower() == model_answer.strip().lower()


if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description="Evaluate model output.")
  parser.add_argument('output_file', type=Path, help="Path to the output file.")
  parser.add_argument('--comp_answer', type=str, default="comp_answer_basic", help="Comparison function name.")
  parser.add_argument('--filter', type=str, default="yes_no_filter", help="Filter function name (optional).")
  args = parser.parse_args()
  
  comp_answer = globals().get(args.comp_answer)
  filter_func = globals().get(args.filter, None)
  if comp_answer is None:
    raise ValueError(f"Comparison function '{args.comp_answer}' not found.")
  if filter_func is None:
    logger.warning(f"Filter function '{args.filter}' not found, using no filter.")
  
  evaluate(args.output_file, comp_answer)