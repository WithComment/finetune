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
source ~/venv/finetune/bin/activate

dataset_name=$1
system_prompt=$2
checkpoint_name=$3
npus=${4-1}
# Add system prompt argument only if system_prompt is not empty
if [[ -n "${system_prompt}" ]]; then
    system_arg="--system ${HOME}/finetune/qwenvl/data/prompts/${system_prompt}.txt"
else
    system_arg=""
fi

if [[ "${checkpoint_name}" == */last ]]; then
    checkpoint_stem="${checkpoint_name%/last}"
else
    checkpoint_stem="${checkpoint_name}"
fi

NPROC_PER_NODE=${npus} \
CUDA_VISIBLE_DEVICES=$(seq 0 $((npus - 1)) | paste -sd, -) \
MAX_PIXELS=1003520 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
LOG_LEVEL='INFO' \
FPS=1 \
swift infer \
    --use_hf True \
    --model "${CHECKPOINT_DIR}/${checkpoint_name}" \
    --model_type qwen2_5_vl \
    --infer_backend pt \
    --truncation_strategy None \
    --max_length 17000 \
    --val_dataset "withcomment/${dataset_name}" \
    --max_batch_size 1 \
    --val_dataset_shuffle True \
    --max_new_tokens 32 \
    "${system_arg}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --dataloader_num_workers 8 \
    --dataset_num_proc 60 \
    --result_path "${HOME}/finetune/results/${dataset_name}/${checkpoint_stem}/${system_prompt}/results.jsonl" \
    --ignore_args_error True \
    --predict_with_generate True \
    --remove_unused_columns False \
    --seed 903 \
    --full_determinism True \
    --remove_unused_columns False \
    --temperature 0