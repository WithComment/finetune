#!/bin/bash
# filepath: /projects/cft_vlm/Qwen2.5-VL/qwen-vl-finetune/scripts/sft_slurm.sh

#SBATCH --job-name=qwen2_5vl_infer
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --array=0-10%1
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/infer/%j.out
#SBATCH --error=logs/infer/%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

source /fs01/projects/cft_vlm/.venv/bin/activate

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NPROC_PER_NODE=4
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

dataset_use=$1
model_use=$2
if [ -z "$model_use" ]; then
  echo "No model specified, using default: 'Qwen/Qwen2.5-VL-3B-Instruct'"
  dataset_use="Qwen/Qwen2.5-VL-3B-Instruct"
fi

eval_args="
  --model_name_or_path ${model_use}"

data_args="
    --dataset_use ${dataset_use} \
    --split test"

proc_args=""

args="
    ${eval_args} \
    ${data_args} \
    ${proc_args}"

torchrun --nproc_per_node=$NPROC_PER_NODE \
              --nnodes=1 \
              -m qwenvl.predict ${args}