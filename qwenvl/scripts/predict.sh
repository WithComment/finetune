#!/bin/bash
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

# Default values
dataset_use=""
sys_prompt="default"
requeue="false"
model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct"
split="test"
portion="1.0"
use_chat_template="True"
# ...existing code...
rst_prompt=""

# Parse positional and keyword arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --dataset_use)
            if [[ $# -lt 2 ]]; then
                echo "Error: --dataset_use requires a value"
                exit 1
            fi
            dataset_use="$2"
            shift 2
            ;;
        --sys_prompt)
            if [[ $# -lt 2 ]]; then
                echo "Error: --sys_prompt requires a value"
                exit 1
            fi
            sys_prompt="$2"
            shift 2
            ;;
        --rst_prompt)
            if [[ $# -lt 2 ]]; then
                echo "Error: --rst_prompt requires a value"
                exit 1
            fi
            rst_prompt="$2"
            shift 2
            ;;
        --requeue)
            if [[ $# -lt 2 ]]; then
                echo "Error: --requeue requires a value"
                exit 1
            fi
            requeue="$2"
            shift 2
            ;;
        --model_name_or_path)
            if [[ $# -lt 2 ]]; then
                echo "Error: --model_name_or_path requires a value"
                exit 1
            fi
            model_name_or_path="$2"
            shift 2
            ;;
        --split)
            if [[ $# -lt 2 ]]; then
                echo "Error: --split requires a value"
                exit 1
            fi
            split="$2"
            shift 2
            ;;
        --portion)
            if [[ $# -lt 2 ]]; then
                echo "Error: --portion requires a value"
                exit 1
            fi
            portion="$2"
            shift 2
            ;;
        --use_chat_template)
            if [[ $# -lt 2 ]]; then
                echo "Error: --use_chat_template requires a value"
                exit 1
            fi
            use_chat_template="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "${dataset_use}" ]]; then
    echo "Error: --dataset_use is required."
    exit 1
fi

echo "Debug: dataset_use='$dataset_use'"
echo "Debug: sys_prompt='$sys_prompt'"
echo "Debug: rst_prompt='$rst_prompt'"
echo "Debug: requeue='$requeue'"
echo "Debug: model_name_or_path='$model_name_or_path'"
echo "Debug: split='$split'"
echo "Debug: portion='$portion'"
echo "Debug: use_chat_template='$use_chat_template'"

model_args="
  --model_name_or_path ${model_name_or_path}"

data_args="
    --dataset_use ${dataset_use} \
    --split ${split} \
    --portion ${portion} \
    --eval_batch_size 4 \
    --model_max_length 8192"

proc_args="
    --use_chat_template ${use_chat_template} \
    --add_generation_prompt True"

if [[ -n "${rst_prompt}" ]]; then
    proc_args="${proc_args} --rst_prompt ${rst_prompt}"
fi

if [[ -n "${sys_prompt}" ]]; then
    proc_args="${proc_args} --sys_prompt ${sys_prompt}"
fi

args="
    ${model_args} \
    ${data_args} \
    ${proc_args}"
# ...existing code...

echo "Starting evaluation process in the background..."
torchrun --nnodes=1 --nproc_per_node=4 -m qwenvl.predict ${args} &
PROC_ID=$!

echo "Waiting for process $PROC_ID. The script can now receive signals."
wait $PROC_ID
EXIT_CODE=$?

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