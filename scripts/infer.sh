export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export NPROC_PER_NODE=4
export CUDA_HOME=/pkgs/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


torchrun --nproc_per_node=$NPROC_PER_NODE \
              --nnodes=1 \
              --master_addr=$MASTER_ADDR \
              --master_port=$MASTER_PORT \
              qwenvl/test/infer.py \
              --model_name "Qwen/Qwen2.5-VL-${1}B-Instruct" \
              --benchmark "vqa-rad/yes-no"