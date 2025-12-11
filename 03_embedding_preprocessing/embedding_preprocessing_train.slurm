#!/bin/bash
#SBATCH --job-name=embedding_preprocessing_train
#SBATCH --output=embedding_preprocessing_train.%j.log
#SBATCH --error=embedding_preprocessing_train.%j.log
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "$WORK/conda/miniconda3/etc/profile.d/conda.sh"
conda activate embedding_preprocessing

echo "Starting Python script..."
python -u embedding_preprocessing_train.py
echo "Job finished at $(date)"