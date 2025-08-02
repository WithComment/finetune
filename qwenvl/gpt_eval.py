import json
from pathlib import Path
import time
from pydantic import BaseModel
from google import genai
from google.genai import types
from qwenvl.utils import get_logger

logger = get_logger(__name__)


class OnePrediction(BaseModel):
  id: int
  question: str
  answer: str
  model_answer: str
  is_correct: bool


def gpt_eval(result_path: Path, model: str = "gemini-2.5-pro") -> Path:
  prompt = """You are an expert evaluator tasked with assessing whether a model's answer to a visual question is correct.

You will be given a JOSN file. Each object in the file contains four fields:

```json
[
  {{
    "id": <id of question>,
    "question": <A question about visual content>,
    "answer": <the ground truth (correct) answer>,
    "model_answer": <The model's generated answer>
  }}
]
```

Your job is to determine if the `model_answer` is **semantically** correct, even if the wording differs from the ground truth.
**DO NOT MODIFY EXISTING FIELDS**.
Add a 5th field, `is_correct` to the new JOSN objects with the value true if the model's answer is correct, and false if it is not.
Make sure that the value of the is_correct field of each object is based solely on data in the object, and not on any other objects in the file.

{file_content}
"""
  prompt = prompt.format(file_content=result_path.read_text())
  client = genai.Client()
  logger.info(f"Evaluating results using model: {model}")

  start_time = time.time()
  logger.info(f"Starting evaluation at {start_time}")
  response = client.models.generate_content(
      model=model,
      contents=[prompt],
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema=list[OnePrediction],
          temperature=0,
      )
  )
  end_time = time.time()

  logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
  eval_result_path = result_path.with_name(
      f"{model.replace('-', '_')}_results.json")
  with open(eval_result_path, "w") as f:
    f.write(response.text)
  logger.info(f"Eval results saved to {eval_result_path}")
  return eval_result_path


def get_summary(eval_result_path: Path, summary_path: Path, isvalid: bool):
  total = 0
  correct = 0
  invalid = 0
  with open(eval_result_path, "r") as f:
    responses = json.load(f)
  for obj in responses:
    total += 1
    is_correct = obj.get("is_correct")
    if not isinstance(is_correct, bool):
      invalid += 1
    if is_correct is True:
      correct += 1
  accuracy = correct / total if total > 0 else 0
  valid = total - invalid
  accuracy_without_invalid = correct / valid if valid > 0 else 0
  summary = {
      "is_valid": isvalid,
      "model": eval_result_path.stem,
      "total": total,
      "correct": correct,
      "invalid": invalid,
      "accuracy": accuracy,
      "accuracy_without_invalid": accuracy_without_invalid
  }
  with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
  logger.info(f"Summary saved to {summary_path}")
  logger.info(summary)
  return summary


def assert_validity(result_path: Path, eval_result_path: Path):
  eval_results = json.load(open(eval_result_path, "r"))
  results = json.load(open(result_path, "r"))
  isvalid = True
  for og, gemini in zip(results, eval_results):
    try:
      assert og["id"] == gemini["id"]
      assert og["question"] == gemini["question"]
      assert og["answer"] == gemini["answer"]
      assert og["model_answer"] == gemini["model_answer"]
    except AssertionError as e:
      logger.info(og)
      isvalid = False
      break
  if not isvalid:
    logger.error(
      f"Validation failed: {result_path} and {eval_result_path} do not match.")
  return isvalid
