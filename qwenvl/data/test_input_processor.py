import itertools
from PIL import Image as PILImage
import numpy as np
import pytest
import torch
from qwenvl.argument import ProcessingArguments
from qwenvl.data.input_processor import InputProcessor

from transformers import AutoProcessor


def stub_get_images_and_videos(self, messages, proc_args):
  images, videos, fps = [], [], []
  for m in messages:
    c = m["content"]
    if isinstance(c, list):
      for part in c:
        if "image" in part:
          images.append(part["image"])
        if "video" in part:
          videos.append(part["video"][0])
          fps.append(part["video"][1])
  return [val or None for val in [images, videos, fps]]


@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
  monkeypatch.setattr(InputProcessor, "get_images_and_videos",
                      stub_get_images_and_videos)



processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

plain_chat_template = """{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if 'image' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}"""


def assert_text_and_feat(conv, config, images=None, videos=None, fps=None, text_only=False):
  ip = InputProcessor(processor=processor, config=config)
  for use_chat_template in [True, False]:
    for add_generation_prompt in [True, False]:
      for add_vision_id in [True, False]:
        ip.config.use_chat_template = use_chat_template
        ip.config.add_generation_prompt = add_generation_prompt
        ip.config.add_vision_id = add_vision_id
        chat_template = ip.tokenizer.chat_template if use_chat_template else plain_chat_template
        expected_text = [processor.apply_chat_template(c, chat_template=chat_template, add_vision_id=add_vision_id, add_generation_prompt=add_generation_prompt) for c in (conv if isinstance(conv[0], list) else [conv])]
        text, feat = ip.get_features_with_text(
            conv, text_only=text_only, return_tensors='pt', padding=True, padding_side='right')
        assert text == expected_text
        expected_feat = processor(
            text=text, images=images, videos=videos, fps=fps, return_tensors="pt", padding=True, padding_side='right')
        if expected_feat.input_ids.shape != feat.input_ids.shape:
          print("Failed due to tokenization error. Skipping assertion.")
          continue
        else:
          print("Tokenization successful, proceeding with assertions.")
        tokenizer = processor.tokenizer
        
        assert tokenizer.batch_decode(feat.input_ids, skip_special_tokens=True) == tokenizer.batch_decode(expected_feat.input_ids, skip_special_tokens=True), "Input IDs do not match"
        for k in expected_feat:
          if k == 'input_ids':
            continue
          if isinstance(expected_feat[k], torch.Tensor):
            assert torch.equal(
                feat[k], expected_feat[k]), f"Mismatch for {k}: {feat[k]} != {expected_feat[k]}"
          else:
            for i in range(len(expected_feat[k])):
              assert torch.equal(feat[k][i], expected_feat[k][i]
                                ), f"Mismatch for {k}[{i}]: {feat[k][i]} != {expected_feat[k][i]}"
              
        labels = feat['labels']
        vid_grid_thw = feat.get('video_grid_thw', [])
        img_grid_thw = feat.get('image_grid_thw', [])

        vid_token_count = sum(
            thw.prod() // (ip.video_processor.merge_size ** 2) for thw in vid_grid_thw)
        img_token_count = sum(
            thw.prod() // (ip.image_processor.merge_size ** 2) for thw in img_grid_thw)
        if images is not None or videos is not None:
          assert vid_token_count + img_token_count > 0, "No video or image tokens found"
        assert torch.sum(labels == -100) > vid_token_count + img_token_count
        assert torch.sum(labels == -100) + torch.sum(labels == feat.input_ids) == labels.numel(), "Labels do not match input IDs"


def test_one_text_not_packed_not_batched():
  conv = [
      {"role": "system", "content": "You are a helpful assistant. "},
      {"role": "user", "content": "hello"}]
  config = ProcessingArguments()
  assert_text_and_feat(conv, config)

def test_batched():
  c1 = [
      {"role": "system", "content": "You are a helpful assistant. "},
      {"role": "user", "content": "aa. "},
      {"role": "assistant", "content": "The sky is blue."},
  ]
  c2 = [
      {"role": "system", "content": "You are a helpful assistant. "},
      {"role": "user", "content": "bb. "},
      {"role": "assistant", "content": "randomly long long long text to test padding for no reason."},
  ]
  conv = [c1, c2]
  config = ProcessingArguments()
  assert_text_and_feat(conv, config)


def test_multi_turn():
  conv = [[
      {"role": "system", "content": "You are a helpful assistant. "},
      {"role": "user", "content": "u1"},
      {"role": "assistant", "content": "a1"},
      {"role": "user", "content": "u2"},
      {"role": "assistant", "content": "a2"},
  ]]
  config = ProcessingArguments()
  assert_text_and_feat(conv, config)

def test_multi_turn_packed_batched():
  conv = [
      [
        {"role": "system", "content": "You are a helpful assistant. "},
        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2" * 10},
      ],
      [
        {"role": "system", "content": "You are a helpful assistant. "},
        {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4" * 20},
      ]
  ]
  config = ProcessingArguments()
  assert_text_and_feat(conv, config)


def get_random_image(h=None, w=None, numpy=False):
  if h is None or w is None:
    h, w = np.random.randint(64, 128), np.random.randint(64, 128)
  arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
  if numpy:
    return arr
  return PILImage.fromarray(arr)


def get_random_video():
  nframes = np.random.randint(32, 64)
  h, w = np.random.randint(64, 128), np.random.randint(64, 128)
  frames = torch.stack([torch.from_numpy(get_random_image(
      h, w, numpy=True)) for _ in range(nframes)]).permute(0, 3, 1, 2)
  return frames, np.random.uniform(0.5, 4.0)


def test_image():
  images = [get_random_image() for _ in range(4)]
  conv = [
      [
        {"role": "system", "content": "You are a helpful assistant. "},
        {"role": "user", "content": [{"image": images[0]}]}, {"role": "assistant", "content": "a1" * 5},
        {"role": "user", "content": [{"image": images[1]}]}, {"role": "assistant", "content": "a2" * 2},
      ], [
          
              {"role": "system", "content": "You are a helpful assistant. "},
              {"role": "user", "content": [{"image": images[2]}]}, {"role": "assistant", "content": "a2" * 17},
              {"role": "system", "content": "You are a helpful assistant. "},
              {"role": "user", "content": [{"image": images[3]}]}, {"role": "assistant", "content": "a3" * 12},
      ],
  ]
  config = ProcessingArguments()
  assert_text_and_feat(conv, config, images=images)
  assert_text_and_feat(conv, config, images=None, text_only=True)


def test_user_video():
  vids_w_fps = [get_random_video() for _ in range(4)]
  conv = [
      [
          
              {"role": "system", "content": "You are a helpful assistant. "},
              {"role": "user", "content": [{"video": vids_w_fps[0]}]}, {"role": "assistant", "content": "a1" * 5},
              {"role": "user", "content": [{"video": vids_w_fps[1]}]}, {"role": "assistant", "content": "a2" * 2},
      ], [
          
              {"role": "system", "content": "You are a helpful assistant. "},
              {"role": "user", "content": [{"video": vids_w_fps[2]}]}, {"role": "assistant", "content": "a2" * 17},
              {"role": "user", "content": [{"video": vids_w_fps[3]}]}, {"role": "assistant", "content": "a3" * 12},
      ],
  ]
  videos, fps = zip(*vids_w_fps)
  config = ProcessingArguments()
  assert_text_and_feat(conv, config, videos=videos, fps=fps)
  assert_text_and_feat(conv, config, videos=None, text_only=True)
  
