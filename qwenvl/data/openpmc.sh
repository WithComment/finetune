#!/bin/bash

#SBATCH --job-name=openpmc-attach-num-tokens
#SBATCH --nodes=1
#SBATCH --partition cpu
#SBATCH -c 64
#SBATCH --mem=64G
#SBATCH --qos=m2
#SBATCH --time=4:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err

source /projects/cft_vlm/.venv/bin/activate
cd /projects/cft_vlm/finetune
export PYTHONPATH="/projects/cft_vlm/finetune:$PYTHONPATH"
python qwenvl/data/OpenpmcDataset.py $1