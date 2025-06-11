#!/bin/bash
# filepath: /projects/cft_vlm/Qwen2.5-VL/qwen-vl-finetune/scripts/sft_slurm.sh

#SBATCH --job-name=qwen2_5vl-sft
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/sft/%j.out
#SBATCH --error=logs/sft/%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

source /fs01/projects/cft_vlm/.venv/bin/activate

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NPROC_PER_NODE=$SLURM_NTASKS_PER_NODE
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

dataset_use=$1

base_model=Qwen/Qwen2.5-VL-3B-Instruct
run_name="${base_model}-${dataset_use}"
output_dir="/projects/cft_vlm/.checkpoint/${run_name}"

# Create output directory
mkdir -p ${output_dir}

# Training arguments
model_args="
    --model_name_or_path ${base_model} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True"

data_args="
    --dataset_use ${dataset_use} \
    --data_packing True \
    --use_cft False \
    --split train \
    --model_max_length 3072 \
    --num_proc 24"

proc_args=""

train_args="
    --deepspeed /projects/cft_vlm/finetune/qwenvl/scripts/zero3.json \
    --optim adamw_torch \
    --output_dir ${output_dir} \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-7 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8
    --model_max_length 3072 \
    --run_name ${run_name} \
    --report_to wandb"

args=" \
    ${model_args} \
    ${data_args} \
    ${proc_args} \
    ${train_args}"

python -m qwenvl.data.count_tokens ${model_args} ${data_args} ${proc_args}

torchrun --nnodes=1 --nproc_per_node=4 \
    -m qwenvl.train ${args} 