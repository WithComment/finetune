import re
import json
from typing import Optional, Dict, Any


class ToolCallParser:
  """Handles parsing of tool calls from generated text"""

  @staticmethod
  def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the generated text to detect tool calls
    Expected format: <tool_call>{"name": "function_name", "arguments": {...}}</tool_call>
    """
    pattern = r'<tool_call>(.*?)</tool_call>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
      try:
        tool_call_data = json.loads(match.group(1))
        return {
            'name': tool_call_data.get('name'),
            'arguments': tool_call_data.get('arguments', {}),
            'id': f"call_{hash(match.group(1)) % 10000}",
            'raw_text': match.group(1)
        }
      except json.JSONDecodeError:
        return None

    return None

  @staticmethod
  def format_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Format a tool call for inclusion in text"""
    tool_call_data = {'name': name, 'arguments': arguments}
    return f"<tool_call>{json.dumps(tool_call_data)}</tool_call>"

  @staticmethod
  def format_tool_result(tool_call_id: str, result: str) -> str:
    """Format a tool result for inclusion in text"""
    return f"<tool_result id='{tool_call_id}'>{result}</tool_result>"

  @staticmethod
  def extract_final_response(text: str) -> str:
    """Extract the final response after all tool calls"""
    # Remove all tool calls and results from the text
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'<tool_result.*?>.*?</tool_result>',
                  '', text, flags=re.DOTALL)
    return text.strip()
