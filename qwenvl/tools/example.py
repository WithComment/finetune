import torch
from transformers import AutoProcessor
from qwen_tool_model import Qwen2_5_VLForConditionalGenerationWithTools


def example_weather_tool(location: str) -> str:
  """Example weather tool implementation"""
  # This would typically call a real weather API
  return f"The weather in {location} is sunny with 22°C"


def main():
  # Load model and processor
  model = Qwen2_5_VLForConditionalGenerationWithTools.from_pretrained(
      "Qwen/Qwen2.5-VL-7B-Instruct"
  )
  processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

  # Define tools
  tools = [
      {
          "type": "function",
          "function": {
              "name": "get_weather",
              "description": "Get weather information for a location",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "location": {
                          "type": "string",
                          "description": "The location to get weather for"
                      }
                  },
                  "required": ["location"]
              }
          }
      },
      {
          "type": "function",
          "function": {
              "name": "calculate",
              "description": "Perform mathematical calculations",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "expression": {
                          "type": "string",
                          "description": "Mathematical expression to evaluate"
                      }
                  },
                  "required": ["expression"]
              }
          }
      }
  ]

  # Prepare input
  messages = [
      {
          "role": "user",
          "content": "What's the weather like in New York? Also, what's 15 * 23?"
      }
  ]

  text = processor.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
  )

  inputs = processor(text=[text], return_tensors="pt")

  # Generate with tool calling
  with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        tools=tools,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        max_tool_calls=5
    )

  # Decode response
  response = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
  print("Model response:")
  print(response)


if __name__ == "__main__":
  main()
