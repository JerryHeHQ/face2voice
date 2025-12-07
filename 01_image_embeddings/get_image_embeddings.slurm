#!/bin/bash
#SBATCH --job-name=get_image_embeddings
#SBATCH --output=get_image_embeddings.%j.log
#SBATCH --error=get_image_embeddings.%j.log
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "$WORK/conda/miniconda3/etc/profile.d/conda.sh"
conda activate get-image-embeddings

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Starting Python script..."
python -u get_image_embeddings.py
echo "Job finished at $(date)"
