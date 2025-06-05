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

dataset_use=openpmc_tiny

base_model=Qwen/Qwen2.5-VL-3B-Instruct
run_name="${base_model}-${dataset_use//,/_}-$(date +%Y%m%d-%H%M%S)"
output_dir="/projects/cft_vlm/.checkpoint/${base_model}-${dataset_use//,/_}"
cache_dir=/projects/cft_vlm/.cache/training

# Create output directory
mkdir -p ${output_dir}
mkdir -p ${cache_dir}

# Training arguments
args="
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${base_model} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --dataset_use ${dataset_use} \
    --data_packing True \
    --cache_dir ${cache_dir} \
    --optim adamw_torch \
    --model_max_length 512 \
    --output_dir ${output_dir} \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 2e-7 \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training with srun (SLURM's torchrun equivalent)
torchrun --nproc_per_node=${NPROC_PER_NODE} \
              --master_addr=${MASTER_ADDR} \
              --master_port=${MASTER_PORT} \
              qwenvl/train/train_qwen.py ${args}


torchrun --nproc_per_node=$NPROC_PER_NODE \
              --nnodes=1 \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              qwenvl/test/infer.py \
              --model_path ${output_dir} \
              --benchmark "vqa-rad/yes-no"