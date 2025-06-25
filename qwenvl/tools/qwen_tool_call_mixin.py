import torch
from typing import Dict, Any, List, Optional, Union
from .manager import ToolManager
from .parser import ToolCallParser
from .context import ConversationContextManager


class QwenToolCallingMixin:
  """Mixin class to add tool calling capabilities to Qwen2.5-VL models"""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.context_manager = None
    self.tool_parser = ToolCallParser()

  def generate(self, *args, **kwargs):
    """Override generate method to support tool calling"""
    # Extract tool-related parameters
    tools = kwargs.pop('tools', None)
    tool_choice = kwargs.pop('tool_choice', 'auto')
    max_tool_calls = kwargs.pop('max_tool_calls', 10)

    if tools is None:
      # Fall back to standard generation
      return super().generate(*args, **kwargs)

    # Tool calling generation
    return self._generate_with_tools(
        *args,
        tools=tools,
        tool_choice=tool_choice,
        max_tool_calls=max_tool_calls,
        **kwargs
    )

  def _generate_with_tools(
      self,
      input_ids: torch.Tensor,
      tools: List[Dict[str, Any]],
      tool_choice: str = 'auto',
      max_tool_calls: int = 10,
      **kwargs
  ) -> torch.Tensor:
    """
    Generate with tool calling capability
    """
    # Initialize components
    tool_manager = ToolManager(tools)
    self.context_manager = ConversationContextManager(self.processor)

    # Add tools description to the beginning if needed
    if 'attention_mask' in kwargs:
      # Add tool description to the prompt
      tools_prompt = tool_manager.format_tools_for_prompt()
      if tools_prompt:
        tools_tokens = self.processor.tokenizer.encode(
            tools_prompt,
            add_special_tokens=False,
            return_tensors="pt"
        )
        input_ids = torch.cat([tools_tokens, input_ids], dim=1)

        # Update attention mask
        if kwargs['attention_mask'] is not None:
          kwargs['attention_mask'] = self.context_manager.update_attention_mask(
              tools_tokens.new_ones(
                  (tools_tokens.shape[0], tools_tokens.shape[1])),
              kwargs['attention_mask'].shape[1]
          )
          kwargs['attention_mask'] = torch.cat([
              kwargs['attention_mask'][:, :tools_tokens.shape[1]],
              kwargs['attention_mask']
          ], dim=1)

    current_input_ids = input_ids
    tool_calls_made = 0

    # Generation loop
    while tool_calls_made < max_tool_calls:
      # Generate next tokens
      generation_kwargs = self.context_manager.update_generation_kwargs(
          kwargs, current_input_ids.shape[1])
      outputs = super().generate(current_input_ids, **generation_kwargs)

      # Decode generated text
      generated_text = self.processor.tokenizer.decode(
          outputs[0], skip_special_tokens=True)

      # Check if model wants to call a tool
      tool_call = self.tool_parser.parse_tool_call(generated_text)

      if tool_call is None:
        # No tool call, return final response
        return outputs

      # Execute tool call
      tool_result = tool_manager.execute_tool(tool_call)
      tool_calls_made += 1

      # Append tool result to conversation and continue generation
      current_input_ids = self.context_manager.append_tool_result(
          outputs, tool_call, tool_result
      )

      # Update attention mask if provided
      if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
        tool_result_length = current_input_ids.shape[1] - outputs.shape[1]
        kwargs['attention_mask'] = self.context_manager.update_attention_mask(
            kwargs['attention_mask'], tool_result_length
        )

    # If we've reached max tool calls, generate final response
    final_kwargs = self.context_manager.update_generation_kwargs(
        kwargs, current_input_ids.shape[1])
    return super().generate(current_input_ids, **final_kwargs)

  def _format_tools_for_prompt(self, tools: List[Dict[str, Any]]) -> str:
    """Format tools for inclusion in the system prompt (deprecated - use ToolManager)"""
    tool_manager = ToolManager(tools)
    return tool_manager.format_tools_for_prompt()
