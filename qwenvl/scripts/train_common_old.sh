#!/bin/bash
#SBATCH --job-name=sft_qwen2.5-vl
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --qos=m
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/sft/%j.out
#SBATCH --error=logs/sft/%j.err
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --signal=B:TERM@60

# Common setup function
setup_environment() {
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
}

# Common model arguments
get_model_args() {
    local base_model=${1:-"Qwen/Qwen2.5-VL-3B-Instruct"}
    echo "
    --model_name_or_path ${base_model} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True"
}

# Common data arguments
get_data_args() {
    local dataset_use=$1
    local mode=$2
    echo "
    --dataset_use ${dataset_use} \
    --data_packing True \
    --mode ${mode} \
    --split train \
    --model_max_length 8192 \
    --num_proc 24 \
    --force False"
}

# Base training arguments
get_base_train_args() {
    local output_dir=$1
    local run_name=$2
    echo "
    --deepspeed /projects/cft_vlm/finetune/qwenvl/scripts/zero3.json \
    --optim adamw_bnb_8bit \
    --output_dir ${output_dir} \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr_ratio 0.1 \
    --save_strategy steps \
    --save_steps 0.2 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1 \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"
}

# Common training execution
run_training() {
    local dataset_use=$1
    local mode=$2
    local requeue=$3
    local base_model=$4

    local run_name="${base_model}-${dataset_use}-${mode}"
    local output_dir="/projects/cft_vlm/.checkpoint/${run_name}"
    
    # Create output directory
    mkdir -p ${output_dir}
    
    # Get all arguments
    local model_args=$(get_model_args "$base_model")
    local data_args=$(get_data_args "$dataset_use" "$mode")
    local proc_args=""
    local train_args=$(get_base_train_args "$output_dir" "$run_name")
    
    local args=" \
        ${model_args} \
        ${data_args} \
        ${proc_args} \
        ${train_args}"
    
    # Count tokens
    python -m qwenvl.data.count_tokens ${model_args} ${data_args} ${proc_args}
    
    if [ $? -ne 0 ]; then
        echo "Token counting failed. Exiting."
        exit 1
    fi
    
    echo "Starting training process in the background..."
    # Run torchrun in the background and save its PID
    torchrun --nnodes=1 --nproc_per_node=4 -m qwenvl.train ${args} &
    PROC_ID=$!
    
    # Wait for process and handle exit
    echo "Waiting for process $PROC_ID. The script can now receive signals."
    wait $PROC_ID
    EXIT_CODE=$?
    
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
}

setup_environment

dataset_use=$1
mode=${2}
requeue=${3:-true}
base_model=${4:-"Qwen/Qwen2.5-VL-3B-Instruct"}

run_training "$dataset_use" "$mode" "$requeue" "$base_model"