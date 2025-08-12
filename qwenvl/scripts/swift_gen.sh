#!/bin/bash
#SBATCH --job-name=cft_vlm_infer
#SBATCH -A aip-rahulgk
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/infer/%j/%N.log
#SBATCH --error=logs/infer/%j/%N.err
#SBATCH --open-mode=append


source ~/.bashrc
module load cuda
cd ~/finetune

dataset_name=$1
system_prompt=$2
checkpoint_name=$3
checkpoint_dir="${SCRATCH}/checkpoints"

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
LOG_LEVEL='INFO' \
FPS=1 \
swift infer \
    --use_hf True \
    --model "${checkpoint_dir}/${checkpoint_name}" \
    --infer_backend pt \
    --truncation_strategy None \
    --val_dataset "withcomment/${dataset_name}" \
    --max_batch_size 1 \
    --dataset_shuffle false \
    --max_new_tokens 32 \
    --system "${HOME}/finetune/qwenvl/data/prompts/${system_prompt}.txt" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_length 8196 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 60 \
    --result_path "${HOME}/finetune/results/${dataset_name}/${checkpoint_name}/${system_prompt}/results.jsonl" \
    --ignore_args_error True \
    --predict_with_generate True \
    --remove_unused_columns False \
    --model_type qwen2_5_vl