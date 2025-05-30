#!/bin/bash
# filepath: /projects/cft_vlm/Qwen2.5-VL/qwen-vl-finetune/scripts/sft_slurm.sh

#SBATCH --job-name=qwen2vl-sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --qos=m2
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust based on your cluster)
# module load cuda/11.8
# module load python/3.9

# Activate your environment
# source activate your_env_name

# Set distributed training environment variables
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NPROC_PER_NODE=4
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify CUDA is accessible
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA version: $(nvcc --version)"
echo "Available GPUs: $(nvidia-smi -L)"


# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct

# Training hyperparameters
lr=2e-7
batch_size=1  # Reduced for 4 GPUs
grad_accum_steps=8  # Increased to maintain effective batch size

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration - update this with your actual dataset
datasets=openbiomedvid  # Your dataset name

# Output configuration
run_name="qwen2vl-openbiomedvid-$(date +%Y%m%d_%H%M%S)"
output_dir=/projects/cft_vlm/.checkpoint/${run_name}
cache_dir=/projects/cft_vlm/.cache/training

# Create output directory
mkdir -p ${output_dir}
mkdir -p ${cache_dir}

# Training arguments
args="
    --dataset_packing True \
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 0.5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --video_max_frame_pixels 16128 \
    --video_min_frame_pixels 4032 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training with srun (SLURM's torchrun equivalent)
srun torchrun --nproc_per_node=${NPROC_PER_NODE} \
              --master_addr=${MASTER_ADDR} \
              --master_port=${MASTER_PORT} \
              ${entry_file} ${args}