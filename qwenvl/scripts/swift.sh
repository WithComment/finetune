#!/bin/bash
#SBATCH --job-name=cft_vlm_train
#SBATCH -A aip-rahulgk
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=logs/train/%j/%N.log
#SBATCH --error=logs/train/%j/%N.err
#SBATCH --open-mode=append


source ~/.bashrc
module load cuda
cd ~/finetune

# export TRITON_CACHE_DIR=/dev/shm/triton_cache
# mkdir -p /dev/shm/triton_cache

# Accept dataset_name and system_prompt as command-line arguments
dataset_name=$1
system_prompt=$2

mkdir -p /dev/shm/triton_cache

TRITON_CACHE_DIR=/dev/shm/triton_cache \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
LOG_LEVEL='INFO' \
swift sft \
    --use_hf True \
    --model /model-weights/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --freeze_aligner False \
    --system "${HOME}/finetune/qwenvl/data/prompts/${system_prompt}.txt" \
    --dataset "withcomment/swift_${dataset_name}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --packing True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --save_steps 0.5 \
    --save_strategy 'steps' \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 8196 \
    --output_dir "${SCRATCH}/checkpoints/Qwen2_5_3_${dataset_name}_${system_prompt}" \
    --create_checkpoint_symlink True \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 60 \
    --deepspeed zero3 \
    --report_to wandb \
    --add_version False \
    --seed 903