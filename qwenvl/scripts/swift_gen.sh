source ~/.bashrc
module load cuda
cd ~/finetune

dataset_name=$1
system_prompt=$2
checkpoint_path="/model-weights/Qwen2.5-VL-3B-Instruct"

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
LOG_LEVEL='INFO' \
swift infer \
    --use_hf True \
    --model "${checkpoint_path}" \
    --infer_backend pt \
    --truncation_strategy None \
    --val_dataset "withcomment/swift_${dataset_name}" \
    --max_batch_size 1 \
    --dataset_shuffle false \
    --max_new_tokens 32 \
    --system "${HOME}/finetune/qwenvl/data/prompts/${system_prompt}.txt" \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --max_length 8196 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 60 \
    --result_path "${HOME}/finetune/results/${dataset_name}/$(basename "${checkpoint_path}")_${system_prompt}.json" \
    --ignore_args_error True \
    --predict_with_generate True \
    --remove_unused_columns False