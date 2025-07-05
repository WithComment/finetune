from typing import List, Literal
import copy
import torch.nn.functional as F
import itertools

import torch
from qwenvl.argument import ProcessingArguments

from transformers import AutoProcessor, BatchFeature
from transformers import ProcessorMixin, BatchEncoding
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessorKwargs

from qwenvl.data.utils import get_images_and_videos
from qwenvl.utils import get_logger

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
VIDEO_PAD = "<|video_pad|>"
IMAGE_PAD = "<|image_pad|>"

logger = get_logger(__name__)

class InputProcessor(ProcessorMixin):

  attributes = ["image_processor", "tokenizer", "video_processor"]
  valid_kwargs = ["chat_template"]

  image_processor_class = "AutoImageProcessor"
  video_processor_class = "AutoVideoProcessor"
  tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

  def __init__(
      self,
      processor,
      config: ProcessingArguments,
      **kwargs
  ):
    """
    Initialize the input processor for vision-language model fine-tuning.
    Args:
      processor: The main processor containing tokenizer, image_processor, and video_processor.
      chat_template (optional): Template for formatting chat conversations.
      **kwargs: Additional keyword arguments including:
        sys_prompt (str, optional): System prompt for the assistant. Defaults to "You are a helpful assistant."
        mode (str, optional): Processing mode, typically "ift" for instruction fine-tuning. Defaults to "ift".
        add_generation_prompt (bool, optional): Whether to add generation prompt to conversations. Defaults to False.
        add_vision_id (bool, optional): Whether to add vision start/end tokens around visual content. Defaults to False.
        ignore_idx (int, optional): Index value to ignore during loss computation. Defaults to -100.
        proc_args (ProcessingArguments, optional): Processing arguments configuration. Defaults to ProcessingArguments().
    Attributes:
      tokenizer: Tokenizer from the processor for text processing.
      image_processor: Image processor for handling image inputs.
      video_processor: Video processor for handling video inputs.
      image_token (str): Token used as placeholder for images.
      video_token (str): Token used as placeholder for videos.
      image_token_id (int): Token ID for image placeholder.
      video_token_id (int): Token ID for video placeholder.
      vs_id (int): Vision start token ID.
      ve_id (int): Vision end token ID.
    """
    tokenizer = copy.deepcopy(processor.tokenizer)
    image_processor = copy.deepcopy(processor.image_processor)
    video_processor = copy.deepcopy(processor.video_processor)
    self.image_token = "<|image_pad|>" if not hasattr(
        tokenizer, "image_token") else tokenizer.image_token
    self.video_token = "<|video_pad|>" if not hasattr(
        tokenizer, "video_token") else tokenizer.video_token
    self.image_token_id = (
        tokenizer.image_token_id
        if getattr(tokenizer, "image_token_id", None)
        else tokenizer.convert_tokens_to_ids(self.image_token)
    )
    self.video_token_id = (
        tokenizer.video_token_id
        if getattr(tokenizer, "video_token_id", None)
        else tokenizer.convert_tokens_to_ids(self.video_token)
    )
    self.config = config
    chat_template = tokenizer.chat_template
    super().__init__(image_processor, tokenizer,
                     video_processor, chat_template=chat_template)

  def _process_vision(
      self,
      images=None,
      videos=None,
      **kwargs
  ) -> BatchFeature:
    output_kwargs = self._merge_kwargs(
        Qwen2_5_VLProcessorKwargs,
        **kwargs
    )
    img_kwargs = output_kwargs['images_kwargs']
    vid_kwargs = output_kwargs['videos_kwargs']
    return_tensors = vid_kwargs.pop("return_tensors", None)
    vid_kwargs.pop("fps", None)
    image_inputs = videos_inputs = {}
    if images is not None:
      image_inputs = self.image_processor(images=images, **img_kwargs)

    if videos is not None:
      videos_inputs = self.video_processor(videos=videos, **vid_kwargs)
      video_grid_thw = videos_inputs["video_grid_thw"]
      fps = kwargs.get("fps", None)

      if isinstance(fps, (int, float)):
        second_per_grid_ts = [
            self.video_processor.temporal_patch_size / fps] * len(video_grid_thw)
      elif hasattr(fps, "__len__") and len(fps) == len(video_grid_thw):
        second_per_grid_ts = [
            self.video_processor.temporal_patch_size / tmp for tmp in fps]
      else:
        raise ValueError(
            f"The length of fps ({len(fps) if hasattr(fps, '__len__') else fps}) must be equal to the length of video_grid_thw ({len(video_grid_thw)}) or fps should be a single number.")
      videos_inputs.update({"second_per_grid_ts": second_per_grid_ts})

    return BatchFeature(data={**image_inputs, **videos_inputs}, tensor_type=return_tensors)

  def get_images_and_videos(self, messages, proc_args=None):
    return [
      val or None
      for val in get_images_and_videos(messages, proc_args or self.proc_args)
    ]

  def _process_one(
      self,
      messages,
      text_only,
      **kwargs,
  ):
    use_chat_template = self.config.use_chat_template
    add_generation_prompt = self.config.add_generation_prompt
    add_vision_id = self.config.add_vision_id
    ignore_idx = self.config.ignore_idx
    
    
    if add_generation_prompt and not use_chat_template:
      logger.warning(
        "`add_generation_prompt` is set to True, but `use_chat_template` is False. `add_generation_prompt` has no effect when `use_chat_template` is False. "
      )
    text = []
    input_ids = []
    labels = []
    img_idx = 0
    vid_idx = 0
    vs_id = self.tokenizer.encode(VISION_START)[0]
    ve_id = self.tokenizer.encode(VISION_END)[0]
    
    if text_only:
      vision_features = BatchFeature()
    else:
      images, videos, fps = self.get_images_and_videos(messages, self.config)

      vision_features = self._process_vision(
          images=images,
          videos=videos,
          fps=fps,
          **kwargs
      )

    def add_text(m, role):
      '''
      Add text to the input sequence and update input_ids and labels.
      labels are set to ignore_idx for non-assistant roles.
      '''
      text.append(m)
      tokenized_m = self.tokenizer.encode(m)
      input_ids.extend(tokenized_m)
      if role == 'assistant':
        labels.extend(tokenized_m)
      else:
        labels.extend([ignore_idx] * len(tokenized_m))

    def add_vision(vision_type, idx):
      if not vision_type in ['image', 'video']:
        raise ValueError("vision_type must be either 'image' or 'video'.")
      m = ''
      if add_vision_id:
        m += f"{'Picture' if vision_type == 'image' else 'Video'} {idx + 1}: "

      m += f"{VISION_START}<|{vision_type}_pad|>{VISION_END}"
      text.append(m)

      if text_only:
        vision_token_count = 1
      else:
        merge_size = getattr(self, f'{vision_type}_processor').merge_size ** 2
        vision_token_count = vision_features[f'{vision_type}_grid_thw'][idx].prod(
        ) // merge_size
      vision_token_id = getattr(self, f'{vision_type}_token_id')
      input_ids.append(vs_id)
      input_ids.extend([vision_token_id] * vision_token_count)
      input_ids.append(ve_id)
      labels.extend([ignore_idx] * (vision_token_count + 2))
      return idx + 1

    for message in messages:

      if not (isinstance(message['content'], list) or isinstance(message['content'], str)):
        raise ValueError("Content must be a string or a list of dictionaries.")

      if use_chat_template:
        add_text(f"{IM_START}{message['role']}\n", None)

      if isinstance(message['content'], str):
        add_text(f"{message['content']}", message['role'])
      else:
        for content in message['content']:
          if 'image' in content:
            img_idx = add_vision('image', img_idx)
          elif 'video' in content:
            vid_idx = add_vision('video', vid_idx)
          elif 'text' in content:
            add_text(content['text'], message['role'])

      if use_chat_template:
        add_text(f"{IM_END}\n", None)
        
    if use_chat_template and add_generation_prompt:
      add_text(f"{IM_START}assistant\n", 'assistant')
      
    text = ''.join(text)
    be = BatchEncoding(data={
        'input_ids': torch.tensor(input_ids),
        'labels': torch.tensor(labels)})
    return text, be, vision_features

  def get_features_with_text(
      self,
      conversations,
      text_only=False,
      **kwargs
  ):
    
    kwargs['return_tensors'] = 'pt'
    output_kwargs = self._merge_kwargs(
        Qwen2_5_VLProcessorKwargs,
        tokenizer_init_kwargs=self.tokenizer.init_kwargs,
        **kwargs,
    )
    batch_text_features = []
    batch_features = BatchFeature()
    batch_text = []

    if isinstance(conversations[0], dict):
      conversations = [conversations]

    for messages in conversations:
      
      text, text_features, vision_features = self._process_one(
          messages=messages,
          text_only=text_only,
          **kwargs
      )
      batch_text_features.append(text_features)
      for k, v in vision_features.items():
        if k not in batch_features:
          batch_features[k] = []
        batch_features[k].append(v)
      batch_text.append(text)

    for k, v in batch_features.items():
      batch_features[k] = torch.cat(v, dim=0)

    target_length = max(
        [len(feat['input_ids']) for feat in batch_text_features]
    )
    input_ids = [feat['input_ids'] for feat in batch_text_features]
    attn_masks = [[1] * len(feat['input_ids']) for feat in batch_text_features]
    labels = [feat['labels'] for feat in batch_text_features]
    input_ids, attn_masks = pad_and_stack_tensors(
        tensors=input_ids,
        target_length=target_length,
        padding_value=self.tokenizer.pad_token_id,
        **output_kwargs['text_kwargs'],
    )
    labels, _ = pad_and_stack_tensors(
        tensors=labels,
        target_length=target_length,
        padding_value=self.config.ignore_idx,
        **output_kwargs['text_kwargs'],
    )
    batch_features['input_ids'] = input_ids
    batch_features['attention_mask'] = attn_masks
    batch_features['labels'] = labels
    return batch_text, batch_features

  def __call__(
      self,
      conversations,
      text_only=False,
      **kwargs
  ):
    return self.get_features_with_text(conversations, text_only, **kwargs)[1]
  
  def get_text(
      self,
      conversations,
      text_only=False,
      **kwargs,
  ):
    return self.get_features_with_text(conversations, text_only, **kwargs)[0]
  
  
  def get_text_length(
      self,
      conversation,
      **kwargs,
  ):
    """
    Get the length of the tokenized text in the conversation.
    """
    if not isinstance(conversation[0], dict):
      raise ValueError("Conversations must .")
    text, batch_features = self.get_features_with_text(
        conversation, text_only=True, **kwargs)
    return batch_features.input_ids.shape[1]


def pad_and_stack_tensors(
    tensors: List[torch.Tensor],
    target_length: int,
    padding_value: int = 0,
    padding_side: Literal["right", "left"] = "right",
    truncation_side: Literal["right", "left"] = "right",
    **kwargs,
) -> torch.Tensor:
  """
  Pads or truncates a list of 1D tensors to a target length and stacks them.

  This function processes a list of 1D tensors to ensure they all have the
  same specified `target_length`. Tensors shorter than the target are padded,
  and tensors longer than the target are truncated. The processed tensors are
  then stacked into a single 2D tensor.

  Args:
      tensors (List[torch.Tensor]):
          A list of 1D PyTorch tensors to process.
      target_length (int):
          The desired final length for each tensor.
      padding_value (int, optional):
          The value to use for padding. Defaults to 0.
      padding_side (Literal["right", "left"], optional):
          The side to add padding to if a tensor is shorter than
          `target_length`. Defaults to "right".
      truncation_side (Literal["right", "left"], optional):
          The side to truncate from if a tensor is longer than
          `target_length`. Defaults to "right".

  Returns:
      torch.Tensor:
          A 2D tensor of shape (len(tensors), target_length) containing the
          processed and stacked tensors. If the input list is empty, an empty
          tensor of shape (0, target_length) is returned.

  Raises:
      ValueError: If `padding_side` or `truncation_side` are not "left" or "right".
  """
  if padding_side not in ["right", "left"]:
    raise ValueError(
        f"padding_side must be 'right' or 'left', but got '{padding_side}'")
  if truncation_side not in ["right", "left"]:
    raise ValueError(
        f"truncation_side must be 'right' or 'left', but got '{truncation_side}'")

  attention_masks = []
  processed_tensors = []
  device = tensors[0].device

  if any(tensor.dim() != 1 for tensor in tensors):
    raise ValueError("All tensors in the input list must be 1D tensors.")

  for tensor in tensors:
    current_len = len(tensor)

    if current_len >= target_length:
      if truncation_side == 'right':
        processed_tensor = tensor[:target_length]
      else:
        processed_tensor = tensor[-target_length:]
      mask = torch.ones(target_length, dtype=torch.long, device=device)

    elif current_len < target_length:
      pad_len = target_length - current_len
      if padding_side == 'right':
        padding = (0, pad_len)
      else:
        padding = (pad_len, 0)

      processed_tensor = F.pad(tensor, padding, "constant", padding_value)
      mask = torch.ones(current_len, dtype=torch.long, device=device)
      mask = F.pad(mask, padding, "constant", 0)

    processed_tensors.append(processed_tensor)
    attention_masks.append(mask)

  stacked_tensors = torch.stack(processed_tensors)
  stacked_masks = torch.stack(attention_masks)

  return stacked_tensors, stacked_masks


if __name__ == '__main__':
  print(InputProcessor(AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")).tokenizer.pad_token_id)