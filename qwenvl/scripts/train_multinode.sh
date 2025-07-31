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

# Common model arguments
get_model_args() {
    local model_name_or_path=$1
    echo "
    --model_name_or_path ${model_name_or_path} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True"
}

get_data_args() {
    local dataset_use=$1
    local packing=$2
    echo "
    --dataset_use ${dataset_use} \
    --packing ${packing} \
    --split train \
    --model_max_length 8000 \
    --portion 1.0"
}

get_proc_args() {
    local cft_prompt=$1
    local use_chat_template=$2
    local args="--use_chat_template ${use_chat_template}"

    # Only add cft_prompt if it's not empty
    if [[ -n "${cft_prompt}" ]]; then
        args="${args} --cft_prompt ${cft_prompt}"
    fi
    if [[ -n "${sys_prompt}" ]]; then
        args="${args} --sys_prompt ${sys_prompt}"
    fi
    if [[ -n "${usr_prompt}" ]]; then
        args="${args} --usr_prompt ${usr_prompt}"
    fi

    echo "${args}"
}

# Base training arguments
get_base_train_args() {
    local output_dir=$1
    local run_name=$2
    echo "
    --optim adamw_bnb_8bit \
    --output_dir ${output_dir} \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy no \
    --lr_scheduler_type cosine_with_min_lr \
    --min_lr_ratio 0.1 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0 \
    --max_grad_norm 1 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 0.5 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --run_name ${run_name} \
    --report_to wandb"
}

# Common training execution

run_training() {
    local dataset_use=$1
    local cft_prompt=$2
    local model_name_or_path=$3
    local packing=$4
    local use_chat_template=$5
    local sys_prompt=$6
    local usr_prompt=$7

    # Compose run_name: stem is dataset_use, append "-cft" if cft_prompt is not empty
    local run_name="${dataset_use}"
    if [[ -n "${cft_prompt}" ]]; then
        run_name="${run_name}-cft"
    fi
    local model_stem=$(basename "${model_name_or_path}")
    run_name="${model_stem}-${run_name}"
    local output_dir="${SCRATCH}/checkpoints/${run_name}"

    # Create output directory
    mkdir -p "${output_dir}"

    # Get all arguments
    local model_args
    model_args=$(get_model_args "${model_name_or_path}")
    local data_args
    data_args=$(get_data_args "${dataset_use}" "${packing}")
    local proc_args
    proc_args=$(get_proc_args "${cft_prompt}" "${use_chat_template}")
    local train_args
    train_args=$(get_base_train_args "${output_dir}" "${run_name}")

    local args=" \
        ${model_args} \
        ${data_args} \
        ${proc_args} \
        ${train_args}"

    echo "Starting training process..."
    srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=123 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT -m qwenvl.train ${args}

    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "Training crashed with exit code $EXIT_CODE."
    else
        echo "Training completed successfully."
    fi
}

setup_environment


# Parse positional and keyword arguments
dataset_use=""
cft_prompt=""
usr_prompt=""
sys_prompt=""
# lower case for bash.
model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct"
# Must be uppercase for python.
packing="True"
use_chat_template="True"

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
        --cft_prompt)
            if [[ $# -lt 2 ]]; then
                echo "Error: --cft_prompt requires a value"
                exit 1
            fi
            cft_prompt="$2"
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
        --model_name_or_path)
            if [[ $# -lt 2 ]]; then
                echo "Error: --model_name_or_path requires a value"
                exit 1
            fi
            model_name_or_path="$2"
            shift 2
            ;;
        --packing)
            if [[ $# -lt 2 ]]; then
                echo "Error: --packing requires a value"
                exit 1
            fi
            packing="$2"
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
echo "Debug: cft_prompt='$cft_prompt'"
echo "Debug: model_name_or_path='$model_name_or_path'"
echo "Debug: packing='$packing'"
echo "Debug: use_chat_template='$use_chat_template'"
echo "Debug: sys_prompt='$sys_prompt'"
echo "Debug: usr_prompt='$usr_prompt'"

run_training "$dataset_use" "$cft_prompt" "$model_name_or_path" "$packing" "$use_chat_template" "$sys_prompt" "$usr_prompt"
