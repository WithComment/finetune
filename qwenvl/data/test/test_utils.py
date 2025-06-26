from qwenvl.data.utils import make_prompt


def test_make_prompt():
  conversations = [
      [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "Describe this image and video."},
                  {"type": "image", "image": "/path/to/image.jpg"},
                  {"type": "video", "video": "/path/to/video.mp4"}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": "This is a test response."}
              ]
          }
      ]
  ]

  # Test with use_cft=True
  result_cft = make_prompt(
      conversations=conversations,
      tokenizer=None,
      for_training=True,
      use_cft=True
  )

  expected_cft = (
      "Describe this image and video."
      "<|vision_start|><|image_pad|><|vision_end|>"
      "<|vision_start|><|video_pad|><|vision_end|>"
      "This is a test response."
  )

  assert result_cft == expected_cft


if __name__ == "__main__":
  import pytest
  pytest.main(["-v", __file__])