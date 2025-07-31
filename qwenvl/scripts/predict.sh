#!/bin/bash
#SBATCH --job-name=cft_vlm_infer
#SBATCH -A aip-rahulgk
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/infer/%j/%N.log
#SBATCH --error=logs/infer/%j/%N.err
#SBATCH --open-mode=append


module load cuda
source ~/venv/finetune/bin/activate
cd ~/finetune
setup_environment() {

    module load cuda
    source ~/venv/finetune/bin/activate
    cd ~/finetune

    # Set distributed training environment variables
    MASTER_ADDR=$(getent hosts $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) | awk '{ print $1 }')
    MASTER_PORT=29500
    NPROC_PER_NODE=$SLURM_GPUS_ON_NODE

    # Sanity checking
    echo "Master host: $MASTER_HOSTNAME"
    echo "MASTER_ADDR=$MASTER_ADDR"
    echo "MASTER_PORT=$MASTER_PORT"
    echo "Node ID: $SLURM_NODEID"

    # Set env variables
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
}

# Default values
dataset_use=""
sys_prompt="default"
requeue="false"
model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct"
split="test"
portion="1.0"
use_chat_template="True"
# ...existing code...
usr_prompt="default"

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
        --usr_prompt)
            if [[ $# -lt 2 ]]; then
                echo "Error: --usr_prompt requires a value"
                exit 1
            fi
            usr_prompt="$2"
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
echo "Debug: usr_prompt='$usr_prompt'"
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
    --eval_batch_size 1 \
    --model_max_length 16384"

proc_args="
    --use_chat_template ${use_chat_template} \
    --add_generation_prompt True"

if [[ -n "${usr_prompt}" ]]; then
    proc_args="${proc_args} --usr_prompt ${usr_prompt}"
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
torchrun --nnodes=1 --nproc_per_node=4 -m qwenvl.predict ${args}
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "Prediction failed with exit code $EXIT_CODE, not requeuing job."
    exit 1
else
    echo "Prediction completed successfully."
fi