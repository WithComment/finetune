import json
import hashlib
from typing import Any


class ToolManager:
  """Manages tool definitions and execution for Qwen2.5-VL models"""

  def __init__(self, tools: list[dict[str, Any]]):
    self.tools = {tool['function']['name']: tool for tool in tools}

  def execute_tool(self, tool_call: dict[str, Any]) -> str:
    """
    Execute a tool call and return the result
    """
    tool_name = tool_call['name']

    if tool_name not in self.tools:
      return f"Error: Tool '{tool_name}' not found"

    tool_function = self.tools[tool_name]['function']

    try:
      # Here you would implement the actual tool execution
      # This is a placeholder - you need to implement actual tool calling
      result = self._call_function(tool_function, tool_call['arguments'])
      return str(result)
    except Exception as e:
      return f"Error executing tool: {str(e)}"

  def _call_function(self, function_def: dict[str, Any], arguments: dict[str, Any]) -> Any:
    """
    Placeholder for actual function calling implementation
    You need to implement this based on your tool execution framework
    """
    # This could call external APIs, run local functions, etc.
    # For now, return a placeholder response
    return f"Tool {function_def['name']} called with args: {arguments}"

  def format_tools_for_prompt(self) -> str:
    """
    Format tools for inclusion in the system prompt
    """
    if not self.tools:
      return ""

    tools_desc = "You have access to the following tools:\n\n"

    for tool in self.tools.values():
      func = tool['function']
      tools_desc += f"- {func['name']}: {func['description']}\n"

      if 'parameters' in func:
        params = func['parameters']
        if 'properties' in params:
          tools_desc += "  Parameters:\n"
          for param_name, param_info in params['properties'].items():
            required = param_name in params.get('required', [])
            req_str = " (required)" if required else " (optional)"
            tools_desc += f"    - {param_name}: {param_info.get('description', '')}{req_str}\n"
      tools_desc += "\n"

    tools_desc += "To call a tool, use this format: <tool_call>{'name': 'tool_name', 'arguments': {...}}</tool_call>\n"
    tools_desc += "You will receive the tool result in this format: <tool_result id='call_id'>result</tool_result>\n\n"

    return tools_desc

  def generate_tool_call_id(self, tool_call_text: str) -> str:
    """Generate a unique ID for a tool call"""
    return f"call_{hash(tool_call_text) % 10000}"
