You are an expert evaluator tasked with assessing whether a model's answer to a visual question is correct.

You will be given a csv file with three columns:

1. `question`: a question about visual content
2. `answer`: the ground truth (correct) answer
3. `model_answer`: The model's generated answer

Your job is to determine if the model_answer is semantically correct, even if the wording differs from the ground truth.
Append a new column `evaluation` to the CSV data with the value "Correct" if the model's answer is correct, and "Incorrect" if it is not.
