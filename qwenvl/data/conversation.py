from abc import ABC, abstractmethod
import itertools
import random
from typing import Any

class ConversationMaker(ABC):
  '''Abstract base class for creating conversation from an item in a dataset.'''

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


class VQAConversationMaker(ConversationMaker):
  def __init__(self, for_training: bool = True):
    '''
    Args:
      for_training: If True, the conversation will be created for training.
                    If False, it will be created for inference.
    '''
    self.for_training = for_training
    
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
    user_content.append({'text': item.get('question', '')})
    assistant_content = item.get('answer', '')
    conv = [{'role': 'user', 'content': user_content}]
    if self.for_training:
      conv.append({'role': 'assistant', 'content': assistant_content})
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
  Adds a system prompt to the conversation.
  
  Args:
    sys_prompt: The system prompt to add.
  '''
  
  def __call__(self, conversation: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    prompt = random.choice(self.prompts)
    conversation[0].insert(self.idx, {'role': 'system', 'content': prompt})
    return conversation
  
  
class AllPromptAdder(ConversationModifier):
  
  def __call__(self, conversation: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    for i in range(len(conversation)):
      cft_prompt = random.choice(self.prompts)
      conversation[i].insert(self.idx, {'role': self.role, 'content': cft_prompt})
    return conversation


class RolePromptAdder(ConversationModifier):
  '''
  Adds a user prompt to the conversation.
  
  Args:
    user_prompt: The user prompt to add.
  '''
  
  def __call__(self, conversation: list[list[dict[str, Any]]]) -> list[list[dict[str, Any]]]:
    for i in range(len(conversation)):
      user_prompt = random.choice(self.prompts)
      for message in conversation[i]:
        if message['role'] == self.role:
          message['content'].insert(self.idx, {'text': user_prompt})
          break
    return conversation

class ConversationProcessor(ConversationMaker):
  """Handles both one item or a list of items."""
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
      raise ValueError(f"Input item must be a list of dictionaries, got {item} instead.")
    
    conversation = [self.maker(i) for i in item]
    for modifier in self.modifiers:
      conversation = modifier(conversation)
    return list(itertools.chain(*conversation)) # Flatten the list of conversations