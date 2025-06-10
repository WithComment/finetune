#!/bin/bash
#SBATCH --partition=cpu
#SBATCH -c 64
#SBATCH --mem=64G
#SBATCH --job-name=verify_videos
#SBATCH --output=logs/verify_videos_%j.out
#SBATCH --error=logs/verify_videos_%j.err
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

date;hostname;pwd


export XDG_RUNTIME_DIR=""
echo $SLURM_SUBMIT_DIR
/projects/cft_vlm/.venv/bin/python -m qwenvl.data.openbiomedvid