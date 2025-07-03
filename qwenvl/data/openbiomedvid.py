import json
import logging
import random
import shutil
import subprocess
from typing import Callable

import datasets
import numpy as np
from transformers import Qwen2_5_VLProcessor

from qwenvl.argument import ProcessingArguments
from qwenvl.data.base import BaseDataset
from qwenvl.data.utils import make_cot, smart_resize, verify_video


from .sft import SFTDataset


VID_PROMPTS = (
    "Please describe the biomedical content shown in this video.",
    "What medical or clinical content can you observe in this video?",
    "Could you explain the medical aspects shown in this footage?",
    "Please provide a description of the medical content demonstrated in this video.",
    "What biomedical information is being presented in this video?",
    "Can you describe the medical content shown in this footage?",
    "Please explain what you observe in this medical video.",
    "What medical or clinical elements are demonstrated in this video?",
    "Could you describe the biomedical content presented here?",
    "Please detail the medical information shown in this video.",
    "What do you observe in this medical footage?",
    "Can you explain the biomedical content demonstrated here?",
    "Please describe what's being shown in this medical video.",
    "What medical content is being presented in this footage?",
    "Could you detail the biomedical aspects shown in this video?",
    "Please explain the medical elements demonstrated here.",
    "What clinical or medical content do you observe in this video?",
    "Can you describe the biomedical information shown in this footage?",
    "Please provide an explanation of the medical content in this video.",
    "What medical or clinical aspects are being demonstrated here?"
)


CAP_CP = (
    # Application of Knowledge
    "Analyze the following surgical video and caption. Your primary focus is to understand how the demonstrated procedures and techniques are applied in a real-world surgical setting to address a specific medical condition. Correlate the visual information with the described surgical actions to build a practical knowledge base.",

    # In-Depth Exploration
    "This is a surgical procedure. Watch the video and read the caption to conduct an in-depth exploration of the surgical techniques, anatomical structures, and instruments shown. Pay close attention to the subtle details and nuances of each action and its context within the broader procedure.",

    # Reflective Thinking
    "After reviewing the surgery video and caption, engage in reflective thinking. Consider the 'why' behind each surgical action and decision. Analyze the procedure's flow, the techniques used, and potential alternative approaches or critical decision points. The goal is to develop a deeper, more considered understanding of the surgical process.",

    # Creative Interpretation
    "Observe the following surgical procedure. While grounding your analysis in the provided video and caption, consider creative interpretations of the techniques shown. Hypothesize how these methods could be adapted, improved, or applied to different but related surgical scenarios. The objective is to foster innovative thinking in the context of surgical practice.",

    # Summarization and Synthesis
    "Your task is to summarize and synthesize the key information from the provided surgery video and caption. Identify the critical steps, the main instruments used, and the primary objective of the procedure. Condense this information into a clear and concise summary that captures the essence of the surgical event.",

    # Focus on Key Concepts
    "Analyze the surgery video and caption with a focus on identifying the key concepts. These may include critical anatomical landmarks, pivotal surgical maneuvers, or the core principles behind the procedure. Your aim is to extract and learn the most essential information that defines this surgical event.",

    # Contextual Understanding
    "As you process the following surgery video and caption, focus on contextual understanding. Consider the patient's condition, the surgical goals, and the overall phase of the operation. Use this context to interpret the actions and decisions demonstrated in the video more accurately.",

    # Critical Analysis
    "Engage in a critical analysis of the surgical procedure presented in the video and caption. Evaluate the effectiveness of the techniques used, identify any potential risks or challenges, and consider the decision-making process of the surgical team. Your goal is to develop a discerning and analytical perspective on the procedure.",

    # Question-Based Learning
    "Adopt a question-based learning approach to this surgery video and caption. As you observe the procedure, generate questions about the techniques, instruments, and decision-making processes. For example: 'Why was this specific instrument chosen?' or 'What are the potential complications of this maneuver?' Use these questions to drive a deeper investigation of the material.",

    # Comparative Learning
    "Engage in comparative learning. Analyze the surgical video and caption and compare the techniques, instruments, and workflow to other similar or alternative surgical procedures you have been trained on. Identify unique aspects, shared principles, and relative advantages or disadvantages to build a comparative understanding."
)

QA_CP = (
    # Application of Knowledge
    "Your objective is to understand the practical application of the surgical procedure shown. Analyze the video and the provided QA pairs to see how the questions and answers illuminate the real-world purpose and effect of each surgical action. Use the QA pairs as a guide to connect theory to practice.",

    # In-Depth Exploration
    "View the surgical video and examine the accompanying QA pairs. Use each question and its corresponding answer as a starting point for an in-depth exploration of the specific moments, techniques, and anatomical structures referenced in the video. Your goal is to build a detailed and comprehensive understanding based on these inquiries.",

    # Reflective Thinking
    "Engage in reflective thinking about the surgical procedure. The provided QA pairs highlight key decision points and rationales. Consider not just the answers, but the reasoning behind them. Why are these questions important? Use the video and QA list to reflect on the surgical team's intentions, choices, and the underlying principles of the operation.",

    # Creative Interpretation
    "Observe the surgical procedure and review the QA pairs. While the answers provide factual information, your task is to engage in creative interpretation. Based on the visual information and the questions asked, brainstorm potential improvements, alternative techniques, or novel applications for the methods shown. Use the QA list to spark new lines of inquiry.",

    # Summarization and Synthesis
    "Your task is to create a holistic understanding of the surgical event. Watch the video and synthesize the information contained within the entire list of QA pairs. Combine the key insights from each question and answer to construct a comprehensive summary that encapsulates the procedure's objectives, critical phases, and significant events.",

    # Focus on Key Concepts
    "The provided QA pairs are designed to highlight the most important aspects of the surgical procedure. Analyze the video by focusing on the elements addressed in the questions. Your goal is to identify and learn the key concepts, critical maneuvers, and essential anatomical knowledge that the QA list emphasizes.",

    # Contextual Understanding
    "Develop a strong contextual understanding of the surgery. Use the questions and answers not just as standalone facts, but as pieces of a larger puzzle. Synthesize the information from the QA list and the video to grasp the overall clinical scenario, the patient's condition, and the ultimate goals of the surgical intervention.",

    # Critical Analysis
    "Perform a critical analysis of the surgery shown. The QA pairs provide specific points of interest. Use these to critically evaluate the effectiveness, precision, and potential risks of the demonstrated techniques. Assess the quality of the surgical work based on the evidence in the video and the points raised in the questions.",

    # Question-Based Learning
    "Adopt a question-based learning mindset. The provided QA list is your guide. As you process the video, actively consider why these specific questions are being asked. Think about what other questions could be relevant. Your aim is to internalize the process of inquiry as a method for effective learning about surgical procedures.",

    # Comparative Learning
    "Engage in comparative learning. Analyze the surgical video and the detailed answers in the QA pairs. Compare the specific techniques, instrument choices, and procedural logic with other surgical methods in your knowledge base. Use the QA list to identify the unique features and differentiating factors of this particular procedure."
)

class OpenbiomedvidDataset(SFTDataset):
  """
  Openbiomedvid dataset for training and evaluation.
  """

  @staticmethod
  def _get_content(item, media_dir):
    texts = list()
    images = list()
    videos = [media_dir / item['video']]

    return texts, images, videos

  def make_conversation(self, bin):
    conversation = list()
    for item in bin:
      conversation.append(
        self._make_conversation(
          item,
          media_dir=self.media_dir,
          mode=self.mode
        )
      )
    return conversation

  @staticmethod
  def _make_conversation(item, media_dir, mode):
    if mode not in {'cft', 'ift'}:
      raise ValueError(f"Unsupported mode: {mode}. Supported modes are 'cft' and 'ift'.")
    _type = item['type']
    if _type not in {'qa_pairs', 'caption'}:
      raise ValueError(f"Unsupported type: {_type}. Supported types are 'qa_pairs' and 'caption'.")
    text = item[_type]
    if not text:
      raise ValueError(f"Item doesn't have '{_type}': {item}")

    conversation = list()
    vid_path = media_dir / item['video']
    if _type == 'qa_pairs':
      if mode == 'cft':
        conversation.append({'role': None, 'content': [{'type': 'text', 'text': random.choice(QA_CP)}]})
      conversation.append({'role': 'user', 'content': [{'type': 'video', 'video': vid_path}]})
      for qa in text:
        conversation.append({
            'role': 'user',
            'content': [{'type': 'text', 'text': qa['question'] + '\n'}]
        })
        conversation.append({
            'role': 'assistant',
            'content': [{'type': 'text', 'text': qa['answer'] + '\n'}]
        })
    elif _type == 'caption':
      if mode == 'ift':
        conversation.append({
            'role': 'user',
            'content': [{'type': 'video', 'video': vid_path}, {'type': 'text', 'text': random.choice(VID_PROMPTS)}]
        })
        conversation.append({
            'role': 'assistant',
            'content': [{'type': 'text', 'text': text}]
        })
      elif mode == 'cft':
        conversation.append({
          'role': None, # No role in CFT or CPT
          'content': [
            {'type': 'text', 'text': random.choice(CAP_CP)},
            {'type': 'video', 'video': vid_path},
            {'type': 'text', 'text': text}
          ]
        })
    return conversation
  
  @staticmethod
  def _get_num_content_tokens(
      ds: datasets.Dataset,
      processor: Qwen2_5_VLProcessor,
      proc_args: ProcessingArguments,
      get_content_fn: Callable
  ):
    """
    Get the number of content tokens.
    Content are defined by the `get_content` method.
    """
    def _get_num_content_tokens(item):
      metas = json.loads(item['video_metas'])
      num_tokens = 0
      for meta in metas:
        nframes, resolution, fps = (
            meta["num_frames"],
            meta["resolution"],
            meta.get("fps", 1),
        )
        # Load resolution
        if isinstance(resolution, (list, tuple)):
          w, h = resolution
        elif isinstance(resolution, dict):
          w, h = resolution["width"], resolution["height"]
          
        w, h, w_tokens, h_tokens = smart_resize(
          w, h,
          processor.video_processor.max_pixels,
          processor.video_processor.min_pixels)
        num_tokens += h_tokens * w_tokens * nframes // processor.video_processor.temporal_patch_size
        
      item['media_count'] = 1
      item['num_content_tokens'] = num_tokens
      return item

    return ds.map(
      _get_num_content_tokens,
      num_proc=BaseDataset.num_proc,
      desc="Counting content tokens",
    )
    
  @staticmethod
  def _preprocess(ds, media_dir, num_proc):
    def _filter(item):
      return bool(item.get('video_metas')) and verify_video(item, media_dir)

    ds = ds.filter(
        _filter,
        num_proc=num_proc,
        desc="Filtering out items with missing videos"
    )
    return ds


logger = logging.getLogger(__name__)
