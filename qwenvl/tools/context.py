import torch
import json
from typing import List, Tuple, Dict, Any, Optional


class ConversationContextManager:
  """Manages conversation context for tool calling"""

  def __init__(self, processor):
    self.processor = processor
    self.tool_calls_history = []

  def append_tool_result(self, input_ids: torch.Tensor, tool_call: Dict[str, Any], tool_result: str) -> torch.Tensor:
    """
    Append tool result to the input sequence
    """
    # Format tool result
    tool_result_text = f"<tool_result id='{tool_call['id']}'>{tool_result}</tool_result>"

    # Tokenize and append
    tool_result_tokens = self.processor.tokenizer.encode(
        tool_result_text,
        add_special_tokens=False,
        return_tensors="pt"
    )

    # Concatenate with existing input_ids
    new_input_ids = torch.cat([input_ids, tool_result_tokens], dim=1)

    # Store in history
    self.tool_calls_history.append((tool_call, tool_result))

    return new_input_ids

  def build_full_context(self, original_input: str, tool_calls_and_results: List[Tuple[Dict[str, Any], str]]) -> torch.Tensor:
    """
    Build the full conversation context including tool calls and results
    """
    context_parts = [original_input]

    for tool_call, result in tool_calls_and_results:
      # Add the tool call
      tool_call_text = f"<tool_call>{json.dumps({'name': tool_call['name'], 'arguments': tool_call['arguments']})}</tool_call>"
      context_parts.append(tool_call_text)

      # Add the tool result
      tool_result_text = f"<tool_result id='{tool_call['id']}'>{result}</tool_result>"
      context_parts.append(tool_result_text)

    full_context = "".join(context_parts)
    return self.processor.tokenizer.encode(
        full_context,
        return_tensors="pt",
        add_special_tokens=False
    )

  def update_attention_mask(self, attention_mask: torch.Tensor, new_tokens_length: int) -> torch.Tensor:
    """
    Update attention mask for newly added tokens
    """
    batch_size = attention_mask.shape[0]
    new_attention = torch.ones(
        (batch_size, new_tokens_length),
        device=attention_mask.device,
        dtype=attention_mask.dtype
    )

    return torch.cat([attention_mask, new_attention], dim=1)

  def manage_kv_cache(self, past_key_values: Optional[Any], new_sequence_length: int) -> Optional[Any]:
    """
    Properly manage the key-value cache when appending tool results
    """
    if past_key_values is None:
      return None

    # For tool calling, we typically want to reset the cache
    # since we're changing the conversation context significantly
    # But you could also try to preserve parts of it for efficiency

    return None  # Force recomputation for now

  def update_generation_kwargs(self, kwargs: Dict[str, Any], new_length: int) -> Dict[str, Any]:
    """
    Update generation parameters for continued generation
    """
    updated_kwargs = kwargs.copy()

    # Update max_length to account for new tokens
    if 'max_length' in updated_kwargs:
      updated_kwargs['max_length'] = max(
          updated_kwargs['max_length'],
          new_length + updated_kwargs.get('max_new_tokens', 100)
      )

    # Reset some parameters that shouldn't carry over
    updated_kwargs.pop('past_key_values', None)

    return updated_kwargs

  def reset_history(self):
    """Reset the tool call history"""
    self.tool_calls_history = []
