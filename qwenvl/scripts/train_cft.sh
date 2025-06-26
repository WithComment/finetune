#!/bin/bash
# filepath: /projects/cft_vlm/Qwen2.5-VL/qwen-vl-finetune/scripts/sft_slurm.sh

#SBATCH --job-name=sft_qwen2.5-vl
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --qos=m2
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/sft/%j.out
#SBATCH --error=logs/sft/%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --signal=B:TERM@60

trap 'echo "[$(date)] SIGNAL $? received, requeueing"; \
     scontrol requeue $SLURM_JOB_ID; \
     exit 1' USR1 TERM

source /fs01/projects/cft_vlm/.venv/bin/activate
cd /fs01/projects/cft_vlm/finetune

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_use=$1
requeue=${2:-true}

base_model=Qwen/Qwen2.5-VL-3B-Instruct
run_name="${base_model}-${dataset_use}"
output_dir="/projects/cft_vlm/.checkpoint/${run_name}-cft"

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
    --use_cot False \
    --use_cft True \
    --split train \
    --model_max_length 16384 \
    --num_proc 24 \
    --force False"

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
    --save_steps 50 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8
    --run_name ${run_name} \
    --report_to wandb"

args=" \
    ${model_args} \
    ${data_args} \
    ${proc_args} \
    ${train_args}"

python -m qwenvl.data.count_tokens ${model_args} ${data_args} ${proc_args}

if [ $? -ne 0 ]; then
    echo "Token counting failed. Exiting."
    exit
fi

data_args="
    --dataset_use ${dataset_use} \
    --data_packing True \
    --use_cot False \
    --use_cft True \
    --split train \
    --model_max_length 16384 \
    --num_proc 24 \
    --force False"

args=" \
    ${model_args} \
    ${data_args} \
    ${proc_args} \
    ${train_args}"

echo "Starting training process in the background..."
# Run torchrun in the background and save its PID
torchrun --nnodes=1 --nproc_per_node=4 -m qwenvl.train ${args} &
PROC_ID=$!

# The shell is now free and can process signals.
# The 'wait' command pauses the script while allowing traps to fire.
echo "Waiting for process $PROC_ID. The script can now receive signals."
wait $PROC_ID
EXIT_CODE=$?

# This part of the script will only be reached if the wait command
# is NOT interrupted by a signal that causes the script to exit.
if [ $EXIT_CODE -ne 0 ]; then
    if [ "$requeue" = false ]; then
        echo "Training failed with exit code $EXIT_CODE, not requeuing job."
        exit 1
    fi
    echo "Training crashed with exit code $EXIT_CODE, resubmitting job."
    scontrol requeue $SLURM_JOB_ID
else
    echo "Training completed successfully."
fi