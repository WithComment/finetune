import json
from typing import Callable
from pathlib import Path
import re

from .utils import get_logger

logger = get_logger(__name__)


def yes_no_filter(answer: list[str]) -> bool:
  yes_no_answers = ['yes', 'no', 'y', 'n']
  return any(c in yes_no_answers for c in answer)

def chexpert_filter(answer: list[str]) -> bool:
  chexpert_answers = [
    'present', 'absent', 'ap', 'pa', 'lateral', 'frontal',
  ]
  return any(c in chexpert_answers for c in answer)

def mc_filter(answer: list[str]) -> bool:
  return any(c in ['a', 'b', 'c', 'd'] for c in answer)


def parse_answer_basic(answer: str, k: int = 3) -> str:
  return re.split(r'[^\w]+', answer.strip().lower())[:k]


def comp_answer_basic(answer: str, model_answer: str) -> bool:
  return answer[0] in model_answer


def evaluate(
    output_file: Path,
    parse_answer: Callable = parse_answer_basic,
    comp_answer: Callable = comp_answer_basic,
    filter: Callable = yes_no_filter
) -> dict:
  items = list()
  with open(output_file, 'r') as f:
    items = [json.loads(l) for l in f]
    
  total = 0
  correct = 0
  incorrect = 0
  invalid = 0
  
  for item in items:
    answer = parse_answer(item['answer'])
    if not filter(answer):
      continue
    total += 1
    model_answer = parse_answer(item['model_answer'])
    if not filter(model_answer):
      invalid += 1
      continue
    if comp_answer(answer, model_answer):
      correct += 1
    else:
      incorrect += 1

  acc_wo_invalid = correct / (total - invalid) if (total - invalid) > 0 else 0
  if total == 0:
    raise ValueError("No items (of the right type) found in the output file.")
  logger.info(f"Total: {total}, Correct: {correct}, Incorrect: {incorrect}, Invalid: {invalid}")
  logger.info(f"Accuracy with invalid: {correct / (total):.2f}")
  logger.info(f"Accuracy without invalid: {acc_wo_invalid:.2f}")
  return {
    'total': total,
    'correct': correct,
    'incorrect': incorrect,
    'invalid': invalid,
    'accuracy_with_invalid': correct / total,
    'accuracy_without_invalid': acc_wo_invalid,
  }


if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description="Evaluate model output.")
  parser.add_argument('output_file', type=Path, help="Path to the output file.")
  parser.add_argument('--parse_answer', type=str, default="parse_answer_basic", help="Answer parsing function name.")
  parser.add_argument('--comp_answer', type=str, default="comp_answer_basic", help="Comparison function name.")
  parser.add_argument('--filter', type=str, default="yes_no_filter", help="Filter function name (optional).")
  args = parser.parse_args()
  
  parse_answer = globals().get(args.parse_answer)
  comp_answer = globals().get(args.comp_answer)
  filter_func = globals().get(args.filter)
  if parse_answer is None:
    raise ValueError(f"Parse answer function '{args.parse_answer}' not found.")
  if comp_answer is None:
    raise ValueError(f"Comparison function '{args.comp_answer}' not found.")
  if filter_func is None:
    logger.warning(f"Filter function '{args.filter}' not found, using no filter.")
  
  summary = evaluate(args.output_file, parse_answer, comp_answer, filter_func)
  output_dir = Path(args.output_file).parent
  with open(output_dir / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
  logger.info(f"Evaluation summary saved to {output_dir / 'summary.json'}")
  logger.info(summary)
