# VLM CFT

## Background

Vision language models often learn shortcuts during SFT. They often generate based on subtle spurious correlation between text tokens, resulting in answers that are not grounded (low fidelity) on the vision inputs. We would like to devise a finetuning strategy that will mitigate this problem. Speficially, we are interested in *Can we improve VLM grounding when we finetune/inference with specific attention guiding prompt?* There are three attention guiding system prompts that we would like to test.
1. **Spatial:** "Focus your attention on the primary subject and any objects it directly interacts with. Isolate key visual details."
2. **Temporal:** "Focus your attention on the sequence of frames showing significant change. Prioritize the frames capturing the action's beginning, peak, and conclusion."
3. **Combination:** "Focus attention on the primary subject's main action. Prioritize the temporal sequence from the action's start to its conclusion, emphasizing the peak. Within this sequence, isolate the key visual details of the subject and its direct interactions."

## Experiments

I have created three sythetic datasets from fashion-MNIST.

The first dataset consists of individual images merged into a 2x2 grid. The question I intend to ask is "What type of clothing is in <> corner of this image?". The goal of this dataset is to test the model's ability to focus specific regions of an image based on spatial text prompts.

The second dataset consists of individual images merged into a video with 10 frames. The question I intend to ask is "What type of clothing is in the <> second of this video?". The goal of this dataset is to test the model's ability to focus specific time interval of a video based on temporal text prompts.

The third dataset is a combination of the first two, where individual images are merged into 2x2 grid, and the grid are merged into videos with 10 frames.

An instantiation of our hypothesis, in the context of the sythetic datasets that I have, would be the model achieve higher accuracy on the spatial dataset when we finetune and then inference using our spatial attention prompt.

I have a reasonably good idea of how to design experiments that will confirm/reject this hypothesis. All we have to do is to take the baseline model, and the model trained on corresponding spatial/temporal system prompts, to test their accuracies on the sythetic spatial/temporal datasets.

What I don't know, however, is how to design further experiments to identify the mechanism behind the improved accuracy. Our hypothesis is that when we finetune with attention guiding prompt, the model learns to attend to vision tokens more. Therefore we should see an increase in attention to vision tokens in that specific region when the model answers the question.

I need you to help me with designing such experiments that will reveal the true mechanism behind the observations. Clearly indicate the purpose of each experiment you design, i.e., state the hypothesis each of them aims to test, and how a positive confirmation of that hypothesis will help with the overall mechanism.
