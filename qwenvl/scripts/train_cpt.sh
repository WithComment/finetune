#!/bin/bash

# Include SLURM directives at the top
#SBATCH --job-name=sft_qwen2.5-vl
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --qos=m2
#SBATCH --gres=gpu:4
#SBATCH --partition=a40
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --output=logs/sft/%j.out
#SBATCH --error=logs/sft/%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@60
#SBATCH --signal=B:TERM@60

# Source common functions
source "$(dirname "$0")/train_common.sh"

# Setup environment
setup_environment

# Get parameters
dataset_use=$1
requeue=${2:-true}

# Run training with CPT mode
run_training "$dataset_use" "cpt" "$requeue"