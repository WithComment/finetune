from dataclasses import dataclass, field
import random
from typing import Optional

from omegaconf import MISSING

from src.data.data_builder import DataBuilder, DataConfig
from datasets import Image, Features, List, Value


@dataclass
class ScienceQAConfig(DataConfig):
  category_filters: Optional[dict[str, list[str]]] = None
  modalities: Optional[list[str]] = None
  prob_with_context: float = MISSING
  prob_with_choices: float = MISSING


class ScienceQABuilder(DataBuilder):
  
  category_filters: dict[str, list[str]] | None
  modalities: list[str] | None
  prob_with_context: float
  prob_with_choices: float
  
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # if self.modalities and 'image' in self.modalities:
    prompt_list = List({'image': Image(decode=True), 'text': Value('string'), 'type': Value('string')})
    # else:
    #   prompt_list = List({'text': Value('string'), 'type': Value('string')})
    self.features = Features({
        'prompt': List({'content': prompt_list, 'role': Value('string')}),
        'completion': List({'content': Value('string'), 'role': Value('string')})
    })
    
  def filter(self, item: dict) -> bool:
    if self.category_filters is not None:
      for key, allowed_values in self.category_filters.items():
        if item[key] not in allowed_values:
          return False

    if self.modalities is not None:
      if item['image'] and 'image' not in self.modalities:
        return False
      if not item['image'] and 'text' not in self.modalities:
        return False

    return True
  
  def input_w_context(self, item: dict) -> list[dict] | dict:
    prompt = []
    for info in self.input_format:
      info = info.lower()
      if info == 'image':
        if item['image']:
          prompt.append({"type": "image", "image": item['image']})
      else:
        prompt.append({"type": "text", "text": f"{info}: {item[info]}\n"})
    return prompt

  def make_choices(self, item: dict) -> tuple[dict, str]:
    choices = item['choices']
    letters = 'ABCDEF'
    permutation = list(range(len(choices)))
    random.shuffle(permutation)
    choices = [choices[i] for i in permutation]
    correct_choice_index = permutation.index(item['answer'])
    
    choice_list = []
    for i, c in enumerate(choices):
      choice_list.append("({}) {}".format(letters[i], c))
    choice_txt = "\n".join(choice_list)
    
    answer = choices[correct_choice_index]
    answer_letter = letters[correct_choice_index]
    return choice_txt, f"{answer_letter}. {answer}"

  def _map(self, item: dict) -> dict:
    
    if random.random() < self.prob_with_context:
      prompt = self.input_w_context(item)
    else:
      prompt = [{"type": "text", "text": item['question']}]
      if item['image']:
        prompt.append({"type": "image", "image": item['image']})
      
    if random.random() < self.prob_with_choices:
      choices, answer = self.make_choices(item)
      prompt.append({"type": "text", "text": f"Options:\n{choices}\n"})
    else:
      answer = item['choices'][item['answer']]
    
    prompt = [{"role": "user", "content": prompt}]
    completion = [{"role": "assistant", "content": answer}]
    return {
        "prompt": prompt,
        "completion": completion
    }
