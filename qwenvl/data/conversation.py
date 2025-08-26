'''
This module defines various classes for creating a conversation from
an item in a dataset in the format required for training.
'''

from abc import ABC, abstractmethod
import itertools
import random
from typing import Any, Callable

from anyio import Path

from .utils import ordinal


class ConversationMaker(ABC):
  '''Abstract base class for creating conversation from an item in a dataset.'''

  def __init__(self, **kwargs):
    '''
    Args:
      for_training: If True, the conversation will be created for training.
                    If False, it will be created for inference.
    '''
    for_training = kwargs.get('for_training', True)
    self.for_training = for_training
    self.media_dir = kwargs.get('media_dir')
    if self.media_dir is not None:
      self.media_dir = Path(self.media_dir)

  @abstractmethod
  def __call__(self, item: list[dict[str, Any]]) -> list[dict[str, Any]]:
    '''Create a conversation from a list of items in a dataset.
    '''
    pass

  def add_media_dir(self, media: str) -> str:
    '''Add media directory to the filename if media_dir is set.'''
    if self.media_dir is not None and isinstance(media, str):
      return str(self.media_dir / media)
    return media

  def get_content(self, item: list[dict[str, Any]]):
    conv = self(item)
    texts, images, videos = [], [], []
    for message in conv:
      content = message['content']
      if isinstance(content, str):
        texts.append(content)
      elif isinstance(content, list):
        for part in content:
          if 'text' in part:
            texts.append(part['text'])
          elif 'image' in part:
            images.append(part['image'])
          elif 'video' in part:
            videos.append(part['video'])
    return texts, images, videos


class TextConversationMaker(ConversationMaker):
  def __init__(self, text_field: str = 'text', **kwargs):
    '''
    Args:
      text_field: The field name in the item that contains the text.
    '''
    super().__init__(**kwargs)
    self.text_field = text_field

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    return [
      # {'role': 'user', 'content': item['cft_prompt']},
      {'role': 'assistant', 'content': item[self.text_field]}
    ]


class VQACM(ConversationMaker):

  def __init__(self, qa_list_field: str = None, q_field: str = 'question', a_field: str = 'answer', **kwargs):
    super().__init__(**kwargs)
    self.qa_list_field = qa_list_field
    self.q_field = q_field
    self.a_field = a_field

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    '''Create a VQA conversation from an item in a dataset.

    Args:
      item: A dictionary representing an item in the dataset.

    Returns:
      A dictionary containing the conversation.
    '''
    user_content = []
    if 'image' in item and 'video' in item:
      raise ValueError("Item cannot contain both 'image' and 'video'.")

    if 'image' in item:
      user_content.append({'image': self.add_media_dir(item['image'])})
    elif 'video' in item:
      user_content.append({'video': self.add_media_dir(item['video'])})
    if self.qa_list_field:
      qa_pairs = item[self.qa_list_field]
    else:
      qa_pairs = [{'question': item[self.q_field], 'answer': item[self.a_field]}]
    
    if isinstance(qa_pairs, dict):
      qa_pairs = [qa_pairs]
      
    conv = [{'role': 'user', 'content': user_content}]
    for qa in qa_pairs:
      conv.append({'role': 'user', 'content': qa[self.q_field]})
      if self.for_training:
        conv.append({'role': 'assistant', 'content': qa[self.a_field]})
    return conv


class MNISTCM(ConversationMaker):
  quadrants = [
    ['top-left', 'top-right'],
    ['bottom-left', 'bottom-right']
  ]
  CHOICES = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress',
    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
  )
  spatial: bool
  temporal: bool
  int2str: Callable[[int], str]
  random: int
  choices: list[str]

  def __init__(self, spatial: bool, temporal: bool, shuffle: int = 1, **kwargs):
    super().__init__(**kwargs)
    self.spatial = spatial
    self.temporal = temporal
    self.random = shuffle
    self.choices = list(self.CHOICES)
    if self.random > 0:
      random.shuffle(self.choices)

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    if self.temporal:
      _type = 'video'
    else:
      _type = 'image'

    media = {_type: self.add_media_dir(item[_type])}

    # loc is (x, y) or (t, x, y).
    loc = item['loc']
    assert (len(loc) == 3 and self.temporal) or (
      len(loc) == 2 and not self.temporal)

    if self.spatial:
      _loc = f'the {self.quadrants[loc[-1]][loc[-2]]} corner '
    else:
      _loc = ''

    if self.temporal:
      _loc += f'during the {ordinal(loc[0] + 1)} second '

    if self.random > 1:
      random.shuffle(self.choices)
    prompt = f"What type of clothing is in {_loc}of this {_type}? Choose exactly one from the following options: "
    prompt += ', '.join(self.choices) + '.\n'
    conv = [{'role': 'user', 'content': [
      media,
      {'text': prompt}
    ]}]
    if self.for_training:
      conv.append(
        {'role': 'assistant', 'content': self.CHOICES[item['label']]})
    return conv


class CaptionCM(ConversationMaker):
  PROMPTS = (
    "Please describe the content shown in the visual content.",
    "What do you observe in the vision input?",
    "Provide a description of the visual input.",
    "Please provide a caption for the vision input.",
  )
  def __init__(self, field_name: str = 'caption', **kwargs):
    '''
    Args:
      field_name: The field name in the item that contains the caption.
    '''
    super().__init__(**kwargs)
    self.field_name = field_name

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    '''Create a caption conversation from an item in a dataset.

    Args:
      item: A dictionary representing an item in the dataset.

    Returns:
      A dictionary containing the conversation.
    '''
    if 'image' in item:
      content = [{'image': item['image']}]
    elif 'video' in item:
      content = [{'video': item['video']}]
    content.append({'text': random.choice(self.PROMPTS)})

    conv = [{'role': 'user', 'content': content}]
    if self.for_training:
      conv.append({'role': 'assistant', 'content': item[self.field_name]})
    return conv


class ClassificationCM(ConversationMaker):
  def __init__(self, exclude_keys: set[str] = None, include_keys: set[str] = None):
    '''
    Args:
      ignore_keys: Theset of keys that are not labels.
    '''
    super().__init__(for_training=True)
    if exclude_keys and include_keys and exclude_keys & include_keys:
      raise ValueError(
        "exclude_keys and include_keys cannot have common elements.")
    self.exclude_keys = exclude_keys
    self.include_keys = include_keys

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    labels = dict()
    for key in item:
      if self.exclude_keys and key in self.exclude_keys:
        continue
      if self.include_keys and key not in self.include_keys:
        continue
      labels[key] = item[key]
    conv = [{'role': 'user', 'content': item['image']}]
    for key, value in labels.items():
      conv.append({'role': 'user', 'content': key.lower() + ': '})
      conv.append({'role': 'assistant', 'content': value})
    return conv


class ChexpertCM(ConversationMaker):
  views = set(['Frontal', 'Lateral', 'AP', 'PA'])

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    questions = item['question']
    answers = item['answer']
    if isinstance(questions, str):
      questions = [questions]
    if isinstance(answers, str):
      answers = [answers]
    if len(questions) != len(answers):
      raise ValueError("Questions and answers must have the same length.")
    img = item['image']
    conv = [{'role': 'user', 'content': [{'image': img}]}]
    for q, a in zip(questions, answers):
      if a in self.views:
        q = f'What is the view of the chest X-ray? Choose from {q}. '
      else:
        q = f'Is {q} present? Choose from present/absent. '
      conv.append({'role': 'user', 'content': q})
      if self.for_training:
        conv.append({'role': 'assistant', 'content': a})
    return conv


class OBVCM(ConversationMaker):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.captionCM = CaptionCM(**kwargs)
    self.VQACM = VQACM(**kwargs)

  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    if item['type'] == 'qa_pairs':
      return self.VQACM(item)
    elif item['type'] == 'caption':
      return self.captionCM(item)


class MCCM(ConversationMaker):
  def __call__(self, item):
    question = "Question: " + item['question']
    for opt, text in item['options'].items():
      question += f"\nOption {opt}: {text}"
    question += "\n"
    conv = [{'role': 'user', 'content': question}]
    return conv


class ConversationModifier(ABC):
  '''
  Abstract base class for modifying conversations.
  For example, adding system prompts, adding CFT prompts, etc.
  Always act on a list of conversations, and return a list of conversations
  '''

  def __init__(self, prompts: list[str], idx: int = 0, role='system', skip_type=[]):
    if isinstance(prompts, str):
      prompts = [prompts]
    self.prompts = prompts
    self.idx = idx
    self.role = role
    self.skip_type = skip_type

  @abstractmethod
  def __call__(self, conversations: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    '''
    Modify the conversation.

    Args:
      conversation: A list of dictionaries representing the conversation.

    Returns:
      A modified conversation.
    '''
    pass


class FirstPromptAdder(ConversationModifier):
  '''
  Add a prompt to the first conversatioin in the pack.

  Args:
    sys_prompt: The system prompt to add.
  '''

  def __call__(self, conversation: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    prompt = random.choice(self.prompts)
    conversation[0].insert(self.idx, {'role': 'system', 'content': prompt})
    return conversation


class AllPromptAdder(ConversationModifier):
  '''Add a prompt to all conversations in the pack.'''

  def __call__(self, conversation: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    for i in range(len(conversation)):
      cft_prompt = random.choice(self.prompts)
      conversation[i].insert(
        self.idx, {'role': self.role, 'content': cft_prompt})
    return conversation


class RolePromptAdder(ConversationModifier):
  '''
  Add a prompt to all messages from a specific role in the conversation.
  Example:
  before: [[{'role': 'user', 'content': ['text': 'What is the capital of France?']}]]
  after: [[{'role': 'user', 'content': [{'text': 'Answer straightforwardly and concisely: '}
    {'text': 'What is the capital of France?'}]},
  '''

  def __call__(self, conversation: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    for i in range(len(conversation)):
      user_prompt = random.choice(self.prompts)
      for message in conversation[i]:
        if message['role'] == self.role:
          if isinstance(message['content'], str):
            message['content'] = [{'text': message['content']}]
          message['content'].insert(self.idx, {'text': user_prompt})
          break
    return conversation


class ConversationProcessor(ConversationMaker):
  """Handles a pack of items and returns flattened conversations."""

  def __init__(
      self,
      conversation_maker: ConversationMaker,
      conversation_modifiers: list[ConversationModifier] = [],
      **kwargs
  ):
    self.maker = conversation_maker
    self.modifiers = conversation_modifiers
    super().__init__(**kwargs)

  @staticmethod
  def merge_messages_from_same_role(conv: list[dict[str, str | list[dict[str, Any]]]]):
    merged = []
    for message in conv:
      if isinstance(message['content'], str):
        message['content'] = [{'text': message['content']}]
      if merged and merged[-1]['role'] == message['role']:
        merged[-1]['content'].extend(message['content'])
      else:
        merged.append(message)
    return merged

  def __call__(self, item: list[dict[str, Any]]) -> list[dict[str, Any]]:
    '''
    Create a conversation from a list of items in a dataset and apply modifiers.

    Args:
      item: A dictionary representing a list of items in a dataset.

    Returns:
      A conversation with modifications such as system prompt.
    '''
    if not isinstance(item, list):
      item = [item]

    conversation = [self.maker(i) for i in item]
    for modifier in self.modifiers:
      if 'type' in item[0] and item[0]['type'] in modifier.skip_type:
        continue
      conversation = modifier(conversation)
    conv = list(itertools.chain(*conversation))
    return self.merge_messages_from_same_role(conv)
