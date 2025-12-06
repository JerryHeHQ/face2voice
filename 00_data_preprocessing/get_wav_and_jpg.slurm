#!/bin/bash
#SBATCH --job-name=get_wav_and_jpg_1
#SBATCH --output=get_wav_and_jpg_1.%j.log
#SBATCH --error=get_wav_and_jpg_1.%j.log
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "$WORK/conda/miniconda3/etc/profile.d/conda.sh"
conda activate get-wav-and-jpg

echo "Starting Python script..."
python -u get_wav_and_jpg.py
echo "Job finished at $(date)"