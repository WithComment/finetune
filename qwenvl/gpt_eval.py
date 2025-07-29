import os
import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from google import genai
from google.genai import types
import logging
from typing import List, Dict, Tuple, Optional, Union
from tqdm.asyncio import tqdm_asyncio
import argparse
import asyncio

# --- Basic Configuration ---
# Configure logging to write to a file and stream to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gemini_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---


def load_json_file(filename: str) -> Union[Dict, List]:
  """
  Load a JSON or JSONL file.

  Args:
      filename (str): Path to the JSON or JSONL file.

  Returns:
      dict or list: The loaded JSON data.
  """
  if not os.path.exists(filename):
    logger.error(f"File not found: {filename}")
    raise FileNotFoundError(f"File not found: {filename}")

  if filename.endswith('.jsonl'):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
      for line in f:
        if line.strip():
          try:
            data.append(json.loads(line.strip()))
          except json.JSONDecodeError:
            logger.warning(
                f"Skipping malformed JSON line in {filename}: {line.strip()}")
    return data
  else:
    with open(filename, 'r', encoding='utf-8') as f:
      return json.load(f)

# --- Main Evaluation Class ---


class GeminiEval:
  def __init__(self, model_name: str, api_keys_path: str = "/h/xiaowenz/api_keys.json"):
    """
    Initialize the Gemini-based evaluation framework.

    Args:
        model_name (str): The name of the Gemini model to use (e.g., 'gemini-1.5-flash').
        api_keys_path (str): Path to the JSON file containing API keys.
    """
    if os.path.exists(api_keys_path):
      with open(api_keys_path, 'r') as f:
        api_keys = json.load(f)
      google_api_key = api_keys.get("gemini")
      if not google_api_key:
        raise ValueError("Google API key not found in api_keys.json")
    else:
      raise FileNotFoundError(f"API keys file not found at: {api_keys_path}")

    # Configure the Generative AI client
    self.model_name = model_name
    self.client = genai.Client(api_key=google_api_key)
    self.results = []
    logger.info(f"GeminiEval initialized with model: {self.model_name}")

  def create_prompt(self, question: str, ground_truth: str, model_answer: str) -> str:
    """Creates the evaluation prompt for a single item."""
    return f"""You are an expert evaluator tasked with assessing whether a model's answer to a visual question is correct. You will be given:

1. A question about visual content
2. The ground truth (correct) answer
3. The model's generated answer

Your job is to determine if the model's answer is semantically correct, even if the wording differs from the ground truth.

Question: {question}
Ground truth: {ground_truth}
Model's answer: {model_answer}

Your response should be "Correct" if the model's answer is correct and "Incorrect" if it is incorrect.
"""

  async def _process_item_async(
      self,
      raw_result: Dict,
      index: int,
      semaphore: asyncio.Semaphore
  ) -> Optional[Dict]:
    """
    Asynchronously processes a single item for evaluation.

    Args:
        raw_result (Dict): The dictionary containing question, answer, etc.
        index (int): The index of the item in the dataset.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrency.

    Returns:
        A dictionary with the processed result or None if an error occurs.
    """
    async with semaphore:
      try:
        question = raw_result['question']
        ground_truth = raw_result['answer']
        model_answer = raw_result['model_answer']

        prompt = self.create_prompt(question, ground_truth, model_answer)

        # Simple retry logic
        for attempt in range(3):
          try:
            response = self.client.models.generate_content(
              model=self.model_name,
              contents=prompt,
              config=types.GenerateContentConfig(
                  thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
              ))
            response_content = response.text.strip()

            result_obj = {
                "question": question,
                "question_id": raw_result.get('question_id', index),
                "question_idx": index,
                "eval_model_id": self.model_name,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "eval_model_response": response_content,
                "custom_id": f"eval_{self.model_name.replace('/', '_')}_{index}",
            }
            return result_obj

          except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} failed for item {index}: {e}")
            if attempt < 2:
              await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
              logger.error(f"Final attempt failed for item {index}. Skipping.")
              return None

      except Exception as e:
        logger.error(
            f"An unexpected error occurred while processing item {index}: {e}")
        return None

  async def run_eval(
      self,
      raw_results: List[Dict],
      output_dir: str,
      concurrency_limit: int = 20
  ):
    """
    Run the evaluation using concurrent requests to the Gemini API.

    Args:
        raw_results (List[Dict]): A list of dictionaries, each with a problem to evaluate.
        output_dir (str): Directory to save the final results.
        concurrency_limit (int): The maximum number of concurrent API requests.
    """
    logger.info(
        f"Starting evaluation for {len(raw_results)} items with concurrency limit {concurrency_limit}.")

    os.makedirs(output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency_limit)

    tasks = [
        self._process_item_async(item, i, semaphore)
        for i, item in enumerate(raw_results)
    ]

    processed_results = await tqdm_asyncio.gather(*tasks)

    # Filter out failed requests (which return None)
    successful_results = [res for res in processed_results if res is not None]
    failed_count = len(raw_results) - len(successful_results)

    self.results = successful_results

    logger.info(
        f"Evaluation complete. Successfully processed: {len(successful_results)}. Failed: {failed_count}.")

    # Save the results
    model_name_sanitized = self.model_name.replace("/", "_")
    output_filename = os.path.join(
        output_dir, f"eval_{model_name_sanitized}_results.jsonl")
    self.save_results(output_filename)

    return self.results

  def save_results(self, output_filepath: str):
    """
    Save all processed results to a JSONL file.

    Args:
        output_filepath (str): The full path to the output file.
    """
    logger.info(f"Saving {len(self.results)} results to {output_filepath}...")
    try:
      with open(output_filepath, "w", encoding='utf-8') as f:
        for result in self.results:
          f.write(json.dumps(result) + '\n')
      logger.info("Results saved successfully.")
    except IOError as e:
      logger.error(f"Failed to save results to {output_filepath}: {e}")


async def main():
  """Main function to run the script."""
  parser = argparse.ArgumentParser(
      description="Evaluation using Gemini models.")
  parser.add_argument("--eval_data_path", type=str, default="/projects/cft_vlm/datasets/surgeryvid/results/test",
                      help="Path to the directory containing model results.")
  parser.add_argument("--models", type=str, default="Qwen2.5-VL-3B-Instruct",
                      help="Name of the model whose outputs are being evaluated.")
  parser.add_argument("--eval_model", type=str, default="gemini-2.5-flash",
                      help="The Gemini model to use for evaluation.")
  parser.add_argument("--output_dir", type=str, default="/projects/cft_vlm/datasets/surgeryvid/results/test",
                      help="Path to the directory to save evaluation results.")
  parser.add_argument("--concurrency", type=int, default=20,
                      help="Number of concurrent requests to the Gemini API.")
  args = parser.parse_args()

  # Initialize the framework
  evaluator = GeminiEval(model_name=args.eval_model)

  raw_results_path = os.path.join(
      args.eval_data_path, args.models, "results.jsonl")
  logger.info(f"Loading raw results from: {raw_results_path}")
  raw_results = load_json_file(raw_results_path)

  # Define the specific output directory for this model's evaluation
  output_dir_for_model = os.path.join(args.output_dir, args.models)

  logger.info(f"🚀 Starting evaluation with {args.eval_model}...")
  await evaluator.run_eval(
      raw_results=raw_results,
      output_dir=output_dir_for_model,
      concurrency_limit=args.concurrency
  )
  logger.info("✅ Evaluation run finished.")


if __name__ == "__main__":
  # The user requested 'gemini-2.5-flash', which may be a future or private model.
  # We will use the provided string. If it's invalid, the API will raise an error.
  # The current public fast model is 'gemini-1.5-flash'.
  # This script will now run asynchronously.
  asyncio.run(main())
