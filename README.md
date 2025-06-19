# Finetuning Qwen2.5-VL

## Repository structure
- `data` contains all classes and utilities used to deal with dataset. They include formatting functions, fetching images/videos, and collate functions.
- `scripts` contains bash scripts for training and inference.

## Inference
Use the following command to perform inference
```sh
bash qwenvl/scripts/predict.sh <dataset_name> <split> <optional: model_checkpoint>
```
For example
```sh
bash qwenvl/scripts/predict.sh vqa-rad test Qwen/Qwen2.5-VL-3B-Instruct-openbiomedvid
```
which will use the `vqa-rad` dataset from huggingface and the checkpoint trained on OpenBiomedVideo dataset.
The output is located at `/projects/cft_vlm/datasets/<dataset_name>/results/<split>/<model_name_final_component>/results.jsonl`.