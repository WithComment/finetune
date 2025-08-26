#!/bin/bash
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 32 
#SBATCH -A aip-rahulgk
#SBATCH --mem=100G
#SBATCH --job-name=alignmeent
#SBATCH --output=%j.log
#SBATCH --time=24:00:00

date;hostname;pwd

module load cuda
source /home/xiaowenz/.bashrc
source /home/xiaowenz/ib
cd /project/6101781/xiaowenz/finetune
source /home/xiaowenz/venv/finetune/bin/activate

python -m experiments.alignment
