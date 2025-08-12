#!/bin/bash
#SBATCH --job-name=cft_vlm_train
#SBATCH -A aip-rahulgk
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train/%j/%N.log
#SBATCH --error=logs/train/%j/%N.err
#SBATCH --open-mode=append


source ~/.bashrc
module load cuda
cd ~/finetune

# Accept dataset_path and system_prompt as command-line arguments
system_prompt=${1:-"default"}
model_path=${2:-"${CHECKPOINT_DIR}/InternVL3_8_Instruct"}

mkdir -p /dev/shm/triton_cache

# Add system prompt argument only if system_prompt is not empty
if [[ -n "${system_prompt}" ]]; then
    system_arg="--system ${HOME}/finetune/qwenvl/data/prompts/${system_prompt}.txt"
else
    system_arg=""
fi

TRITON_CACHE_DIR=/dev/shm/triton_cache \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
LOG_LEVEL='INFO' \
USE_HF='True' \
swift sft \
    --model ${model_path} \
    ${system_arg} \
    --output_dir "${CHECKPOINT_DIR}/$(basename ${model_path})_surgeryvid_train_${system_prompt}" \
    --dataset "withcomment/surgeryvid_train" \
    --use_hf True \
    --model_type internvl3 \
    --train_type full \
    --freeze_aligner False \
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
    --max_length 16384 \
    --create_checkpoint_symlink True \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 60 \
    --deepspeed zero3_offload \
    --report_to wandb \
    --add_version False \
    --seed 903 \
    --full_determinism True \
    --remove_unused_columns False