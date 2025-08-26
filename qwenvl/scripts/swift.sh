#!/usr/bin/env bash
#
# Unified SLURM submit script for Qwen VL fine-tuning.
#
# Usage:
#   ./swift_unified_submit.sh <CONFIG> <dataset> <system_prompt> <model_name> [model_path_override] [eval_flag] [--dry-run] [--time 2-00:00:00]
#
# CONFIG format:
#   <GPU_TYPE>x<GPUS_PER_NODE>[x<NODES>]
#     GPU_TYPE: L40 | H100 (case-insensitive; L40 -> l40s, H100 -> h100/h100s)
#     Example: L40x4        (1 node, 4 L40 GPUs)
#              L40x4x2      (2 nodes, 4 L40 per node)
#              H100x4       (1 node, 4 H100)
#              H100x8       (1 node, 8 H100S style)
#              H100x8x2     (2 nodes, 8 per node)
#
# Positional args (after CONFIG):
#   dataset            e.g. surgeryvid
#   system_prompt      e.g. default (omit .txt)
#   model_name         e.g. Qwen2_5_32
#   model_path_override (optional)
#   eval_flag          true|false (optional; default false)
#
# Environment overrides (optional):
#   PROJECT_ROOT, VENV_PATH, CHECKPOINT_DIR, ACCOUNT, PARTITION, JOB_NAME_PREFIX
#
set -euo pipefail

# ------------- Defaults -------------
PROJECT_ROOT="${PROJECT_ROOT:-/project/6101781/xiaowenz/finetune}"
[ -d "$PROJECT_ROOT" ] || PROJECT_ROOT="${HOME}/finetune"
VENV_PATH="${VENV_PATH:-$HOME/venv/finetune/bin/activate}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$PROJECT_ROOT/output}"
ACCOUNT="${ACCOUNT:-aip-rahulgk}"
PARTITION="${PARTITION:-}"          # leave empty unless needed (adds --partition)
JOB_NAME_PREFIX="${JOB_NAME_PREFIX:-cft_vlm}"
TIME_DEFAULT="2-00:00:00"

DRY_RUN="false"
CONFIG="L40x4"
DATASET="surgeryvid"
SYSTEM_PROMPT="default"
MODEL_NAME="Qwen2_5_7"
EVAL_FLAG="false"
TIME_LIMIT="$TIME_DEFAULT"
PACKING="True"

while [ $# -gt 0 ]; do
  case "$1" in
    --config=*) CONFIG="${1#--config=}"; shift ;;
    --dataset=*) DATASET="${1#--dataset=}"; shift ;;
    --system_prompt=*) SYSTEM_PROMPT="${1#--system_prompt=}"; shift ;;
    --model_name=*) MODEL_NAME="${1#--model_name=}"; shift ;;
    --eval_flag=*) EVAL_FLAG="${1#--eval_flag=}"; shift ;;
    --packing=*) PACKING="${1#--packing=}"; shift ;;
    --dry-run) DRY_RUN="true"; shift ;;
    *)
      echo "Unknown or missing required arg: $1"
      echo "Usage: ./swift_common.sh --config=... --dataset=... --system_prompt=... --model_name=... [--model_path_override=...] [--eval_flag=...] [--dry-run] [--time=2-00:00:00]"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$CONFIG" || -z "$DATASET" || -z "$SYSTEM_PROMPT" || -z "$MODEL_NAME" ]]; then
  echo "ERROR: Missing required arguments."
  echo "Usage: ./swift_common.sh --config=... --dataset=... --system_prompt=... --model_name=... [--model_path_override=...] [--eval_flag=...] [--dry-run] [--time=2-00:00:00]"
  exit 1
fi

# ------------- Config Parsing -------------
# Accept patterns like L40x4x2 / h100x8 / H100x4
IFS='xX' read -r RAW_TYPE GPUS_PER_NODE NODES_OPT <<<"$CONFIG" || {
  echo "Malformed CONFIG: $CONFIG"; exit 1; }

if [[ -z "${GPUS_PER_NODE}" ]]; then
  echo "CONFIG must specify at least GPU type and gpus per node (e.g., L40x4)"; exit 1
fi

if [[ -z "${NODES_OPT:-}" ]]; then
  NODES=1
else
  NODES="$NODES_OPT"
fi

GPU_TYPE_UPPER="$(echo "$RAW_TYPE" | tr '[:lower:]' '[:upper:]')"
case "$GPU_TYPE_UPPER" in
  L40)  GRES_TYPE="l40s"; MEM_PER_GPU=124; CPUS_PER_GPU=16 ;;
  H100) GRES_TYPE="h100"; MEM_PER_GPU=250; CPUS_PER_GPU=6 ;;
  *) echo "Unsupported GPU type: $RAW_TYPE (supported: L40, H100)"; exit 1 ;;
esac

if ! [[ "$GPUS_PER_NODE" =~ ^[0-9]+$ && "$NODES" =~ ^[0-9]+$ ]]; then
  echo "GPUs per node and nodes must be integers"; exit 1
fi

TOTAL_GPUS=$(( GPUS_PER_NODE * NODES ))
MEM_PER_NODE=$(( MEM_PER_GPU * GPUS_PER_NODE ))   # GiB
CPUS_PER_TASK=$(( CPUS_PER_GPU * GPUS_PER_NODE ))
NPROC_PER_NODE="$GPUS_PER_NODE"

# ------------- Paths & Outputs -------------
MODEL_PATH="${CHECKPOINT_DIR}/${MODEL_NAME}"
OUTPUT_DIR_NAME="${MODEL_NAME}_${DATASET}_${SYSTEM_PROMPT}"

SYSTEM_ARG=""
if [[ -n "${SYSTEM_PROMPT}" && "${SYSTEM_PROMPT}" != "none" ]]; then
  PROMPT_FILE="${HOME}/finetune/qwenvl/data/prompts/${SYSTEM_PROMPT}.txt"
  if [[ -f "$PROMPT_FILE" ]]; then
    SYSTEM_ARG="--system $PROMPT_FILE"
  else
    echo "[WARN] System prompt file missing: $PROMPT_FILE (continuing without)"
  fi
fi


PER_DEVICE_BS=1
GRAD_ACCUM=$(( 32 / TOTAL_GPUS / PER_DEVICE_BS ))

TRAIN_ARGS="--model ${MODEL_PATH} \
    ${SYSTEM_ARG} \
    --output_dir ${CHECKPOINT_DIR}/${OUTPUT_DIR_NAME} \
    --dataset withcomment/${DATASET} \
    --dataset_shuffle True \
    --use_hf True \
    --model_type qwen2_5_vl \
    --train_type full \
    --freeze_aligner False \
    --freeze_vit False \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_length 17000 \
    --packing ${PACKING} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${PER_DEVICE_BS} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate 1e-5 \
    --target_modules all-linear \
    --save_steps 0.2 \
    --save_strategy steps \
    --save_total_limit 1 \
    --save_only_model True \
    --logging_steps 1 \
    --create_checkpoint_symlink True \
    --warmup_ratio 0.01 \
    --dataloader_num_workers ${CPUS_PER_GPU} \
    --dataset_num_proc ${CPUS_PER_TASK} \
    --deepspeed zero3_offload \
    --report_to wandb \
    --add_version False \
    --seed 903 \
    --full_determinism True \
    --remove_unused_columns False"

CUDA_VISIBLE_LIST=$(seq 0 $(( GPUS_PER_NODE - 1 )) | paste -sd, -)

# ------------- Build SBATCH Script -------------
JOB_NAME="${JOB_NAME_PREFIX}_${GRES_TYPE}x${GPUS_PER_NODE}x${NODES}_${DATASET}"
STAMP=$(date +%Y%m%d_%H%M%S)
JOB_SCRIPT="${PWD}/swift_job_${JOB_NAME}_${STAMP}.sh"

echo 'a'
if [[ ${NODES} -gt 1 ]]; then
    LAUNCH='
echo "[INFO] Multi-node detected -> torchrun via srun"
MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=29500
echo "[INFO] MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$MASTER_PORT"
srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ms-swift/swift/cli/sft.py \
    $TRAIN_ARGS'
else
    LAUNCH='
echo "[INFO] Single node -> swift sft"
swift sft $TRAIN_ARGS'
fi

if [[ "${EVAL_FLAG}" == "true" ]]; then
    POST_TRAIN="
echo [INFO] Running post-training generation & eval
bash qwenvl/scripts/swift_gen.sh ${DATASET}_test#1000 ${SYSTEM_PROMPT} ${OUTPUT_DIR_NAME}/last ${GPUS_PER_NODE}
# If your eval script path differs adjust below:
if [ -x /home/xiaowenz/finetune/sandbox-ethan/single_eval.sh ]; then
    bash /home/xiaowenz/finetune/sandbox-ethan/single_eval.sh ${DATASET}_test#1000 ${SYSTEM_PROMPT} ${OUTPUT_DIR_NAME}
else
    echo [WARN] No eval script found.
fi
"
else
    POST_TRAIN=''
fi


cat > "$JOB_SCRIPT" <<- EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -N ${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS_PER_TASK}
#SBATCH --gres=gpu:${GRES_TYPE}:${GPUS_PER_NODE}
#SBATCH --mem=${MEM_PER_NODE}G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=logs/train/%j/%N.log
#SBATCH --error=logs/train/%j/%N.err
#SBATCH --open-mode=append
#SBATCH --network=ib0

set -euo pipefail
date; hostname; pwd

export NPROC_PER_NODE=${NPROC_PER_NODE}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_LIST}
export MAX_PIXELS=1003520
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOG_LEVEL=INFO
export TRITON_CACHE_DIR=/dev/shm/triton_cache
export OMP_NUM_THREADS=1
export USE_HF=True
export FPS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export CHECKPOINT_DIR="${CHECKPOINT_DIR}"

module load cuda
source ${HOME}/.bashrc
source ${HOME}/ib
source ${VENV_PATH}
cd ${PROJECT_ROOT}
mkdir -p /dev/shm/triton_cache

echo "[INFO] Job config:"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Nodes: ${NODES}"
echo "  Total GPUs: ${TOTAL_GPUS}"
echo "  GPU Type: ${GRES_TYPE}"
echo "  CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
echo "  Output dir: ${CHECKPOINT_DIR}/${OUTPUT_DIR_NAME}"

TRAIN_ARGS="${TRAIN_ARGS}"
${LAUNCH}
${POST_TRAIN}

echo "[INFO] Job complete."
EOF

chmod +x "$JOB_SCRIPT"

echo "[INFO] Generated job script: $JOB_SCRIPT"
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY-RUN] Not submitting. Use: sbatch $JOB_SCRIPT"
else
  echo "[INFO] Submitting..."
  sbatch "$JOB_SCRIPT"
  rm -f "$JOB_SCRIPT"
fi