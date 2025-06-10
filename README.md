# Finetuning Qwen2.5-VL

## Repository structure
- `data` contains all classes and utilities used to deal with dataset. They include formatting functions, fetching images/videos, and collate functions.
- `scripts` contains bash scripts for training and inference.

## Inference
Use the following command to perform inference
```sh
bash qwenvl/scripts/predict.sh <dataset_name> <optional: model_checkpoint>
```
For example
```sh
bash qwenvl/scripts/predict.sh vqa-rad
```
which will use the `vqa-rad` dataset from huggingface and the original `Qwen2.5-VL-3B-Instruct` model.
The path to the generated output will be shown at the end of the log.