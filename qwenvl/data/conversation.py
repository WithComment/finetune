from abc import ABC, abstractmethod
import itertools
import random
from typing import Any

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

  @abstractmethod
  def __call__(self, item: list[dict[str, Any]]) -> list[dict[str, Any]]:
    '''Create a conversation from a list of items in a dataset.
    '''
    pass
  
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
  
  def __init__(self, qa_list_field: str = None, **kwargs):
    super().__init__(**kwargs)
    if not self.for_training and qa_list_field:
      raise ValueError("For inference, ask questions one by one. That is, do not use qa_list_field.")
    
    self.qa_list_field = qa_list_field
    
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
      user_content.append({'image': item['image']})
    elif 'video' in item:
      user_content.append({'video': item['video']})
    if self.qa_list_field:
      qa_pairs = item[self.qa_list_field]
    else:
      qa_pairs = [{'question': item['question'], 'answer': item['answer']}]
    
    conv = [{'role': 'user', 'content': user_content}]
    for qa in qa_pairs:
      conv.append({'role': 'user', 'content': qa['question']})
      if self.for_training:
        conv.append({'role': 'assistant', 'content': qa['answer']})
    return conv
  
  
class MNISTCM(ConversationMaker):
  def __call__(self, item: dict[str, Any]) -> list[dict[str, Any]]:
    '''Create a MNIST conversation from an item in a dataset.
    
    Args:
      item: A dictionary representing an item in the dataset.
      
    Returns:
      A dictionary containing the conversation.
    '''
    conv = [{'role': 'user', 'content': [
      {'image': item['image']},
      {'text': 'What type of clothing is in this image?'}
    ]}]
    if self.for_training:
      conv.append({'role': 'assistant', 'content': str(item['label'])})
    return conv
  
  
class CaptionCM(ConversationMaker):
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
    if self.field_name not in item:
      raise ValueError(f"Item must contain '{self.field_name}' field.")
    
    conv = [{'role': 'user', 'content': item['image']}]
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
      raise ValueError("exclude_keys and include_keys cannot have common elements.")
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
    self.captionCM = CaptionCM(field_name='caption', for_training=self.for_training)
    self.VQACM = VQACM(qa_list_field='qa_pairs', for_training=self.for_training)
  
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
  def __init__(self, prompts: list[str], idx: int=0, role='system'):
    if isinstance(prompts, str):
      prompts = [prompts]
    self.prompts = prompts
    self.idx = idx
    self.role = role
    
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
      conversation[i].insert(self.idx, {'role': self.role, 'content': cft_prompt})
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
  ):
    self.maker = conversation_maker
    self.modifiers = conversation_modifiers
  
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
      conversation = modifier(conversation)
    return list(itertools.chain(*conversation)) # Flatten the list of conversations