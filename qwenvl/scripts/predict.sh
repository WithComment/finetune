#!/bin/bash
# filepath: /projects/cft_vlm/Qwen2.5-VL/qwen-vl-finetune/scripts/sft_slurm.sh

#SBATCH --job-name=infer-qwen2_5vl
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --qos=m2
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/infer/%j.out
#SBATCH --error=logs/infer/%j.err
#SBATCH --requeue
#SBATCH --time=8:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --signal=B:TERM@60

trap 'echo "[$(date)] SIGNAL $? received, requeueing"; \
     scontrol requeue $SLURM_JOB_ID; \
     exit 1' USR1 TERM

source /fs01/projects/cft_vlm/.venv/bin/activate
cd /fs01/projects/cft_vlm/finetune

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
model_use=${2:-"Qwen/Qwen2.5-VL-3B-Instruct"}
split=${3:-"test"}
portion=${4:-1.0}
sys_prompt=${5:-"default"}
requeue=${6:-false}

model_args="
  --model_name_or_path ${model_use}"

data_args="
    --dataset_use ${dataset_use} \
    --split ${split} \
    --portion ${portion} \
    --eval_batch_size 1"

proc_args="
    --sys_prompt ${sys_prompt} \
    --use_chat_template True \
    --add_generation_prompt True"

args="
    ${model_args} \
    ${data_args} \
    ${proc_args}"


echo "Starting evaluation process in the background..."
# Run torchrun in the background and save its PID
torchrun --nnodes=1 --nproc_per_node=4 -m qwenvl.predict ${args} &
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
        echo "Prediction failed with exit code $EXIT_CODE, not requeuing job."
        exit 1
    fi
    echo "Prediction crashed with exit code $EXIT_CODE, resubmitting job."
    scontrol requeue $SLURM_JOB_ID
else
    echo "Prediction completed successfully."
fi