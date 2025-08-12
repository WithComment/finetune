#!/bin/bash
#SBATCH --job-name=cft_vlm
#SBATCH -A aip-rahulgk
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/train/%j/%N.log
#SBATCH --error=logs/train/%j/%N.err
#SBATCH --open-mode=append
#SBATCH --network=ib0

date;hostname;pwd

# Common setup function
setup_environment() {

    module load cuda
    source ~/venv/finetune/bin/activate
    source ~/ib
    cd ~/finetune

    # Set distributed training environment variables
    MASTER_ADDR=$(getent hosts $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) | awk '{ print $1 }')
    MASTER_PORT=29500
    NPROC_PER_NODE=4

    # Sanity checking
    echo "Master host: $MASTER_HOSTNAME"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "Node ID: $SLURM_NODEID"
    scontrol show hostnames $SLURM_JOB_NODELIST

    # Set env variables
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

dataset_name=$1
system_prompt=$2
num_param=7
mkdir -p /dev/shm/triton_cache
setup_environment

export NPROC_PER_NODE=4
export NNODES=$SLURM_NNODES
export MASTER_PORT=$MASTER_PORT
export MASTER_ADDR=$MASTER_ADDR
export MAX_PIXELS=1003520
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
export LOG_LEVEL='INFO'
export TRITON_CACHE_DIR=/dev/shm/triton_cache
export OMP_NUM_THREADS=1
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ms-swift/swift/cli/sft.py \
    --use_hf True \
    --model "/scratch/xiaowenz/checkpoints/Qwen2_5_${num_param}" \
    --model_type qwen2_5_vl \
    --train_type full \
    --freeze_aligner False \
    --system "${HOME}/finetune/qwenvl/data/prompts/${system_prompt}.txt" \
    --dataset "withcomment/${dataset_name}" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --packing False \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --save_steps 0.5 \
    --save_strategy 'steps' \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 16384 \
    --output_dir "${SCRATCH}/checkpoints/Qwen2_5_${num_param}_${dataset_name}_full_${system_prompt}" \
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