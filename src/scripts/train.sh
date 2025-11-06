#!/bin/bash
#SBATCH --gres=gpu:l40s:4
#SBATCH -c 64
#SBATCH -A aip-rahulgk
#SBATCH --mem=0
#SBATCH --output=logs/%x.log
#SBATCH --time=3:00:00
#SBATCH --exclude=kn003,kn045,kn127,kn027

set -e  # Exit on error

date;hostname;pwd
source ~/.bashrc
source ~/venv/cft/bin/activate
cd ~/finetune

# Store all command line arguments as Hydra overrides
HYDRA_OVERRIDES=("$@")

# Print Hydra overrides if any
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
    echo "Hydra config overrides: ${HYDRA_OVERRIDES[@]}"
fi

accelerate launch --multi_gpu -m src.train.train "${HYDRA_OVERRIDES[@]}"
