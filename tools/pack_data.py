from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import av
import numpy as np
from PIL import Image
from copy import deepcopy
from datasets import Dataset
from transformers import AutoTokenizer, Qwen2VLImageProcessor, Qwen2VLVideoProcessor
# from torchcodec.decoders import VideoDecoder
import binpacking
from tqdm import tqdm
import concurrent.futures
import time

import warnings
warnings.simplefilter('ignore') # In any case, try to avoid warnings as much as possible.

def read_data(file_path):
    """Read JSON or JSONL file"""
    try:
      data = Dataset.from_json(file_path)
      return data.to_list()
    except Exception as e:
      print(e)
      
    if file_path.endswith(('.json', '.jsonl')):
        with open(file_path, 'r') as f:
            if file_path.endswith('.json'):
                return json.load(f)
            return [json.loads(line) for line in f]
    raise ValueError('Please provide a .json or .jsonl file')


def write_data(file_path, data):
    """Write data to JSON or JSONL file"""
    try:
      data = Dataset.from_list(data)
      data.to_json(file_path)
      return
    except Exception as e:
      print('Saving to Dataset failed. Trying to save as JSON or JSONL:', e)
      
    with open(file_path, 'w') as f:
        if file_path.endswith('.json'):
            json.dump(data, f, indent=4)
        elif file_path.endswith('.jsonl'):
            for item in data:
                f.write(json.dumps(item) + '\n')


@dataclass
class DataArguments:
  data_path: str
  max_pixels = 2048 * 28 * 28
  min_pixels = 32 * 28 * 28
  video_max_frame_pixels = 576 * 28 * 28
  video_min_frame_pixels = 144 * 28 * 28
  base_interval = 2
  video_min_frames = 4
  video_max_frames = 8

class MultimodalProcessor:
    def __init__(self, data_args, img_processor, vid_processor, device='cpu'):
        self.data_args = data_args
        self.img_processor = img_processor
        self.vid_processor = vid_processor
        self.device = device

    def _configure_img_processor(self, max_val, min_val):
        processor = deepcopy(self.img_processor)
        processor.max_pixels = max_val
        processor.min_pixels = min_val
        processor.size = {'longest_edge': max_val, 'shortest_edge': min_val}
        return processor
      
    def _configure_vid_processor(self, max_val, min_val):
        processor = deepcopy(self.vid_processor)
        processor.max_frame_pixels = max_val
        processor.min_frame_pixels = min_val
        processor.size = {'longest_edge': max_val, 'shortest_edge': min_val}
        return processor

    def process_image(self, image_file):
        image_path = os.path.join(self.data_args.data_path, image_file)
        if not os.path.exists(image_path):
            print(f'Image file does not exist: {image_path}')
            return 0
        processor = self._configure_img_processor(self.data_args.max_pixels, self.data_args.min_pixels)
        image = Image.open(image_path).convert('RGB')
        visual_processed = processor.preprocess(images=image, return_tensors='pt')
        return visual_processed['image_grid_thw'].prod() // 4

    def process_video(self, video_file):
        video_path = os.path.join(self.data_args.data_path, video_file)
        processor = self._configure_vid_processor(self.data_args.video_max_frame_pixels, self.data_args.video_min_frame_pixels)
        with open(video_path, 'rb') as f:
            vid_data = f.read()
        
        frames = []
        with av.open(BytesIO(vid_data)) as container:
            video_stream = container.streams.video[0]
            total_frames = video_stream.frames
            avg_fps = float(video_stream.average_rate)
            video_length = total_frames / avg_fps
            
            interval = self.data_args.base_interval
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, self.data_args.video_min_frames), self.data_args.video_max_frames)
            
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
            
            for i, frame in enumerate(container.decode(video=0)):
                if i in frame_indices:
                    frames.append(frame.to_rgb().to_ndarray())
                if len(frames) >= target_frames:
                    break
        
        video_frames_numpy = np.array(frames)
        visual_processed = processor.preprocess(images=None, videos=video_frames_numpy, return_tensors='pt')
        return visual_processed['video_grid_thw'].prod() // 4


def calculate_tokens(conversation, processor, tokenizer):
    total_tokens = 21
    roles = {'human': 'user', 'gpt': 'assistant'}
    for message in conversation['conversations']:
        role = message['from']
        text = message['value']
        conv = [{'role': roles[role], 'content': text}]
        encode_id = tokenizer.apply_chat_template(conv, return_tensors='pt', add_generation_prompt=False)[0]
        total_tokens += len(encode_id)
    if 'image' in conversation:
        images = conversation['image'] if isinstance(conversation['image'], list) else [conversation['image']]
        for image_file in images:
            total_tokens += processor.process_image(image_file)
    elif 'video' in conversation:
        videos = conversation['video'] if isinstance(conversation['video'], list) else [conversation['video']]
        for video_file in videos:
            total_tokens += processor.process_video(video_file)
    return total_tokens


def pack_data(data_list, pack_length):
    # Extract the length of each data item
    lengths = [data["num_tokens"] for data in data_list]
    grouped_indices = binpacking.to_constant_volume(
        list(enumerate(lengths)),  # Explicitly convert to list
        pack_length,
        weight_pos=1
    )
    packed_data = []
    for group in grouped_indices:
        group_data = []
        for index, _ in group:
            new_data = data_list[index].copy()
            new_data.pop("num_tokens", None)
            group_data.append(new_data)
        packed_data.append(group_data)
    return packed_data

if __name__ == '__main__':
  import argparse
  argparser = argparse.ArgumentParser()
  argparser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-3B-Instruct', help='Path to the model')
  argparser.add_argument('--data_path', type=str, default='/projects/cft_vlm/openbiomedvid/vid_processed')
  args = argparser.parse_args() 
  
  datasets = {
      'openbiomedvid': {
          'annotation_path': '/projects/cft_vlm/openbiomedvid/processed_for_qwen_finetune.jsonl',
          'data_path': '/projects/cft_vlm/openbiomedvid/vid_processed',
      },
  }

  data_args = DataArguments(data_path=args.data_path)
  
  model_path = args.model_path
  tokenizer = AutoTokenizer.from_pretrained(model_path)
  tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
  img_processor = Qwen2VLImageProcessor.from_pretrained(model_path)
  vid_processor = Qwen2VLVideoProcessor.from_pretrained(model_path)
  print(f'Successfully loaded model components from {model_path}')

  processor = MultimodalProcessor(data_args, img_processor, vid_processor, device='cpu')

  for dataset_name, config in datasets.items():
      processor.data_args.data_path = config['data_path']
      annotation_path = config['annotation_path']
      print(f'\n--- Processing dataset: {dataset_name} ---')
      print(f'Annotation file path: {annotation_path}')
      print(f'Image configuration: max_pixels={data_args.max_pixels}, min_pixels={data_args.min_pixels}')
      print(f'Video frame configuration: video_max_frame_pixels={data_args.video_max_frame_pixels}, video_min_frame_pixels={data_args.video_min_frame_pixels}')
      if not os.path.exists(annotation_path):
          print(f'Annotation file not found: {annotation_path}')
          continue
      data = read_data(annotation_path)
      path_no_suffix = Path(annotation_path).parent / Path(annotation_path).stem
      count_file_path = f'{path_no_suffix}_count.jsonl'
      if os.path.exists(count_file_path):
          print(f"Found pre - calculated token counts, loading data from {count_file_path}.")
          data_with_tokens = read_data(count_file_path)
      else:
          def calculate_and_update(item):
              item['num_tokens'] = calculate_tokens(item, processor, tokenizer)
              return item

          with concurrent.futures.ThreadPoolExecutor() as executor:
              data_with_tokens = list(tqdm(executor.map(calculate_and_update, data), total=len(data), desc=f"Processing {dataset_name} data"))

          # Save the token count results
          write_data(count_file_path, data_with_tokens)
          print(f"Token counts saved to: {count_file_path}")

      # Assume the packing length is 4096
      pack_length = 4096
      # Define the batch size
      batch_size = 256
      all_packed_results = []

      # Record the start time of binpacking
      start_time = time.time()
      for i in range(0, len(data_with_tokens), batch_size):
          batch_data = data_with_tokens[i: i + batch_size]
          batch_packed_result = pack_data(batch_data, pack_length)
          all_packed_results.extend(batch_packed_result)
      # Record the end time of binpacking
      end_time = time.time()

      # Calculate the time spent on binpacking
      binpack_time = end_time - start_time
      print(f"Time spent on binpacking: {binpack_time:.4f} seconds")

      # Save the packed results as a JSON file
      pack_output_path = f'{path_no_suffix}_pack.jsonl'
      write_data(pack_output_path, all_packed_results)
      print(f"Packed results saved to: {pack_output_path}")