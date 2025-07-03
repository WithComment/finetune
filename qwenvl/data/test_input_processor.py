import itertools
from PIL import Image as PILImage
import numpy as np
import pytest
import torch
from qwenvl.input_processor import InputProcessor

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
  return images, videos, fps

@pytest.fixture(autouse=True)
def patch_utils(monkeypatch):
  monkeypatch.setattr(InputProcessor, "get_images_and_videos", stub_get_images_and_videos)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
@pytest.fixture
def ip():
  return InputProcessor(processor)

def assert_text_and_feat(ip, conv, images=None, videos=None, fps=None, packed=False, batched=False):
  if not batched:
    conv = [conv]
  text, feat = ip.get_features_with_text(conv, packed=packed, return_tensors='pt')
  if packed:
    expected_text = [ip.apply_chat_template(list(itertools.chain(*c))) for c in conv]
  else:
    expected_text = [ip.apply_chat_template(c) for c in conv]
    
  assert text == expected_text
  expected_feat = ip.processor(text=text, images=images, videos=videos, fps=fps, return_tensors="pt", padding=True)
  for k in expected_feat:
    if isinstance(expected_feat[k], torch.Tensor):
      assert torch.equal(feat[k], expected_feat[k]), f"Mismatch for {k}: {feat[k]} != {expected_feat[k]}"
    else:
      for i in range(len(expected_feat[k])):
        assert torch.equal(feat[k][i], expected_feat[k][i]), f"Mismatch for {k}[{i}]: {feat[k][i]} != {expected_feat[k][i]}"
  labels = feat['labels']
  vid_grid_thw = feat.get('video_grid_thw', [])
  img_grid_thw = feat.get('image_grid_thw', [])
  
  vid_token_count = sum(thw.prod() // (ip.video_processor.merge_size ** 2) for thw in vid_grid_thw)
  img_token_count = sum(thw.prod() // (ip.image_processor.merge_size ** 2) for thw in img_grid_thw)
  assert torch.sum(labels == -100) > vid_token_count + img_token_count
    
def test_one_text_not_packed_not_batched(ip):
  conv = [{"role": "user", "content": "hello"}]
  assert_text_and_feat(ip, conv, packed=False, batched=False)

def test_packed(ip):
  conv = [[
    {"role": "user", "content": "foo"},
    {"role": "assistant", "content": "bar"},
  ]]
  assert_text_and_feat(ip, conv, packed=True, batched=False)

def test_batched(ip):
  c1 = [
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "A"},
  ]
  c2 = [
    {"role": "user", "content": "bb"},
    {"role": "assistant", "content": "randomly long text to test padding"},
  ]
  assert_text_and_feat(ip, [c1, c2], packed=False, batched=True)


def test_multi_turn_packed(ip):
  conv = [[
    {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"},
    {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"},
  ]]
  assert_text_and_feat(ip, conv, packed=True, batched=False)


def test_multi_turn_packed_batched(ip):
  conv = [
    [
      [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},],
      [{"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2" * 10},]
    ],[
      [{"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},],
      [{"role": "user", "content": "u4"}, {"role": "assistant", "content": "a4" * 20},]
    ],
  ]
  assert_text_and_feat(ip, conv, packed=True, batched=True)


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
  frames = torch.stack([torch.from_numpy(get_random_image(h, w, numpy=True)) for _ in range(nframes)]).permute(0, 3, 1, 2)
  return frames, np.random.uniform(0.5, 4.0)


def test_image(ip):
  images = [get_random_image() for _ in range(4)]
  conv = [
    [
      [{"role": "user", "content": [{"image": images[0]}]}, {"role": "assistant", "content": "a1" * 5},],
      [{"role": "user", "content": [{"image": images[1]}]}, {"role": "assistant", "content": "a2" * 2},]
    ],[
      [{"role": "user", "content": [{"image": images[2]}]}, {"role": "assistant", "content": "a2" * 17},],
      [{"role": "user", "content": [{"image": images[3]}]}, {"role": "assistant", "content": "a3" * 12},]
    ],
  ]
  assert_text_and_feat(ip, conv, images=images, packed=True, batched=True)
  

def test_user_video(ip):
  vids_w_fps = [get_random_video() for _ in range(4)]
  conv = [
    [
      [{"role": "user", "content": [{"video": vids_w_fps[0]}]}, {"role": "assistant", "content": "a1" * 5},],
      [{"role": "user", "content": [{"video": vids_w_fps[1]}]}, {"role": "assistant", "content": "a2" * 2},]
    ],[
      [{"role": "user", "content": [{"video": vids_w_fps[2]}]}, {"role": "assistant", "content": "a2" * 17},],
      [{"role": "user", "content": [{"video": vids_w_fps[3]}]}, {"role": "assistant", "content": "a3" * 12},]
    ],
  ]
  videos, fps = zip(*vids_w_fps)
  assert_text_and_feat(ip, conv, videos=videos, fps=fps, packed=True, batched=True)
