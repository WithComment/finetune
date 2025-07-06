from ..utils import get_logger

logger = get_logger(__name__)

SYS_PROMPTS = {
  'default': "You are a helpful assistant.",
}

CFT_PROMPTS = {
  'path_vqa': [
    # Application of Knowledge
    "Your objective is to understand the clinical significance of the histological findings. Analyze the pathology image and the provided QA pairs to see how the questions and answers connect specific microscopic features to a diagnosis and its real-world clinical implications. Use the QA pairs as a guide to link visual evidence to diagnostic practice.",
    # In-Depth Exploration
    "Examine the pathology image and the accompanying QA pairs. Use each question and its answer as a starting point for an in-depth exploration of the specific regions of interest, morphological features, and tissue structures referenced in the image. Your goal is to build a detailed and comprehensive understanding of the pathology based on these inquiries.",
    # Reflective Thinking
    "Engage in reflective thinking about the diagnostic process. The provided QA pairs highlight key diagnostic criteria and rationales. Consider not just the answers, but the reasoning behind them. Why are these questions important for reaching a diagnosis? Use the image and QA list to reflect on the pathologist's thought process, differential diagnoses, and the underlying principles of histopathology.",
    # Creative Interpretation
    "Observe the pathology image and review the QA pairs. While the answers provide a diagnosis, your task is to engage in diagnostic creativity. Based on the visual features and the questions asked, brainstorm potential differential diagnoses, suggest relevant ancillary tests (like immunohistochemistry or molecular studies), or identify subtle features that might have prognostic value. Use the QA list to spark new lines of inquiry beyond the immediate answer.",
    # Summarization and Synthesis
    "Your task is to create a holistic understanding of the pathology case. Analyze the image and synthesize the information from the entire list of QA pairs. Combine the key insights from each question and answer to construct a comprehensive summary that encapsulates the case's primary findings, critical diagnostic features, and the final conclusion.",
    # Focus on Key Concepts
    "The provided QA pairs are designed to highlight the most important diagnostic aspects of this case. Analyze the pathology image by focusing on the visual elements addressed in the questions. Your goal is to identify and learn the key concepts, critical morphological features, and essential histological knowledge that the QA list emphasizes.",
    # Contextual Understanding
    "Develop a strong contextual understanding of the pathology case. Use the questions and answers not as standalone facts, but as pieces of a larger puzzle. Synthesize the information from the QA list and the image to grasp the overall clinical scenario, the patient's history, and the ultimate goals of the pathological examination, such as diagnosis, grading, or staging.",
    # Critical Analysis
    "Perform a critical analysis of the pathological findings. The QA pairs provide specific points of interest. Use these to critically evaluate the diagnostic significance and potential ambiguity of the features shown. Assess the confidence level of the diagnosis based on the visual evidence in the image and the points raised in the questions, considering potential artifacts or mimics.",
    # Question-Based Learning
    "Adopt a question-based learning mindset. The provided QA list is your guide. As you analyze the image, actively consider why these specific questions are being asked in a diagnostic setting. Think about what other questions a pathologist might ask to arrive at a conclusion. Your aim is to internalize the process of inquiry as a method for effective learning in histopathology.",
    # Comparative Learning
    "Engage in comparative learning. Analyze the pathology image and the detailed answers in the QA pairs. Compare the specific morphological patterns and cellular features with other diseases in your knowledge base, particularly key differential diagnoses. Use the QA list to identify the unique histological features and differentiating factors that confirm this specific diagnosis over others."
  ]
}

avail_datasets = {
  "path_vqa": {
    "ds_dir": "/projects/cft_vlm/datasets/path_vqa/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/path-vqa"
  },
  "vqa_rad": {
    "ds_dir": "/projects/cft_vlm/datasets/vqa_rad/data/dataset",
    "media_dir": None,
    "ds_key": "flaviagiammarino/vqa-rad"
  },
  "slake": {
    "ds_dir": "/projects/cft_vlm/datasets/slake/data/dataset",
    "media_dir": None,
    "ds_key": "mdwiratathya/SLAKE-vqa-english"
  },
  "surgeryvid": {
    "ds_dir": "/projects/cft_vlm/datasets/surgeryvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/surgeryvid/data/vid_processed",
    "ds_key": "withcomment/surgeryvid"
  },
  "open_pmc": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc/data/dataset",
    "media_dir": None,
    "ds_key": "vector-institute/open-pmc"
  },
  "open_pmc_small": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc_small/data/dataset",
    "media_dir": None,
    "ds_key": "vector-institute/open-pmc"
  },
  "open_pmc_tiny": {
    "ds_dir": "/projects/cft_vlm/datasets/open_pmc_tiny/data/dataset",
    "media_dir": None,
    "ds_key": "vector-institute/open-pmc"
  },
  "openbiomedvid": {
    "ds_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/dataset",
    "media_dir": "/projects/cft_vlm/datasets/openbiomedvid/data/vid_processed",
    "ds_key": "withcomment/openbiomedvid"
  }
}
