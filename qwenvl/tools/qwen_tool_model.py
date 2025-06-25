import torch
from typing import Dict, Any, List, Optional, Union
from transformers import Qwen2_5_VLForConditionalGeneration
from .qwen_tool_call_mixin import QwenToolCallingMixin


class Qwen2_5_VLForConditionalGenerationWithTools(QwenToolCallingMixin, Qwen2_5_VLForConditionalGeneration):
  """
  Qwen2.5-VL model with tool calling capabilities

  This class extends the base Qwen2_5_VLForConditionalGeneration with tool calling functionality.
  It can be used as a drop-in replacement for the original model.
  """

  def __init__(self, config, **kwargs):
    super().__init__(config, **kwargs)

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
    """Load a pretrained model with tool calling capabilities"""
    model = super().from_pretrained(
        pretrained_model_name_or_path, *model_args, **kwargs)
    return model

  def prepare_inputs_for_generation(
      self,
      input_ids: torch.LongTensor,
      past_key_values: Optional[torch.Tensor] = None,
      attention_mask: Optional[torch.LongTensor] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      cache_position: Optional[torch.LongTensor] = None,
      **kwargs,
  ) -> Dict[str, Any]:
    """
    Prepare inputs for generation, handling tool calling context
    """
    # Call parent method
    model_inputs = super().prepare_inputs_for_generation(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        **kwargs
    )

    # Handle tool calling specific logic if needed
    if hasattr(self, 'context_manager') and self.context_manager is not None:
      # Apply any tool-specific input modifications
      pass

    return model_inputs
