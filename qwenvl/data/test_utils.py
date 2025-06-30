import torch
from transformers import AutoProcessor
from .utils import make_labels, make_labels_chat, make_prompt


im_start = "<|im_start|>"
im_end = "<|im_end|>"
vision_start = "<|vision_start|>"
vision_end = "<|vision_end|>"
video_pad = "<|video_pad|>"
image_pad = "<|image_pad|>"

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)
tokenizer = processor.tokenizer

def test_make_labels():
  
  cft_cap = f"{im_start}A{vision_start}{video_pad}{vision_end}B{im_end}"
  cft_ids = tokenizer(cft_cap, return_tensors="pt").input_ids
  cft_labels = make_labels(cft_ids, tokenizer)
  assert cft_labels.shape == (1, 7)
  assert torch.all(cft_labels[0, :-2] == -100)
  assert cft_labels[0, -2] == cft_ids[:, 5]
  assert cft_labels[0, -1] == -100
  
def test_make_labels_chat():
  ift_cap = f"{im_start}system\nA{im_end}\n{im_start}user\n{vision_start}{video_pad}{vision_end}B{im_end}\n{im_start}assistant\nC{im_end}"
  ift_ids = tokenizer(ift_cap, return_tensors="pt").input_ids
  ift_labels = make_labels_chat(ift_ids, tokenizer)
  assert torch.all(ift_labels[0, :-2] == -100)
  assert ift_labels[0, -2] == ift_ids[0, -2]
  assert ift_labels[0, -1] == -100
  
if __name__ == "__main__":
  import pytest
  pytest.main([__file__])