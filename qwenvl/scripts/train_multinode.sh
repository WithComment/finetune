#!/bin/bash
#SBATCH --job-name=sft_qwen2.5-vl
#SBATCH --nodes=2
#SBATCH --mem=100GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --qos=m3
#SBATCH -x gpu035
#SBATCH --gres=gpu:8
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/sft/%j.out
#SBATCH --error=logs/sft/%j.err
#SBATCH --requeue
#SBATCH --time=4:00:00
#SBATCH --signal=B:USR1@60
#SBATCH --signal=B:TERM@60

date;hostname;pwd

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
    gpu_count=${SLURM_GPUS_ON_NODE:-$SLURM_NTASKS_PER_NODE}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_CUDA_USE_DSA=1
    export NCCL_IB_DISABLE=1
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
    --model_max_length 8192 \
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
    --save_steps 0.2 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"
}

# Common training execution

run_training() {
    local dataset_use=$1
    local cft_prompt=$2
    local requeue=$3
    local model_name_or_path=$4
    local packing=$5
    local use_chat_template=$6
    local sys_prompt=$7
    local usr_prompt=$8

    # Compose run_name: stem is dataset_use, append "-cft" if cft_prompt is not empty
    local run_name="${dataset_use}"
    if [[ -n "${cft_prompt}" ]]; then
        run_name="${run_name}-cft"
    fi
    local model_stem=$(basename "${model_name_or_path}")
    run_name="${model_stem}-${run_name}"
    local output_dir="/projects/cft_vlm/.checkpoint/${run_name}"

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

    echo "Starting training process in the background..."
    srun -p $SLURM_JOB_PARTITION -c $SLURM_CPUS_ON_NODE -N $SLURM_JOB_NUM_NODES --mem=100GB --gres=gpu:$SLURM_GPUS_ON_NODE torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=$SLURM_GPUS_ON_NODE -m qwenvl.train ${args} &
    PROC_ID=$!

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


# Parse positional and keyword arguments
dataset_use=""
cft_prompt=""
usr_prompt=""
sys_prompt=""
# lower case for bash.
requeue="false"
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
echo "Debug: requeue='$requeue'"
echo "Debug: model_name_or_path='$model_name_or_path'"
echo "Debug: packing='$packing'"
echo "Debug: use_chat_template='$use_chat_template'"
echo "Debug: sys_prompt='$sys_prompt'"
echo "Debug: usr_prompt='$usr_prompt'"

run_training "$dataset_use" "$cft_prompt" "$requeue" "$model_name_or_path" "$packing" "$use_chat_template" "$sys_prompt" "$usr_prompt"