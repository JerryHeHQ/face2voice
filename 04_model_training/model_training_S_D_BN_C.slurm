#!/bin/bash
#SBATCH --job-name=model_training_S_D_BN_C
#SBATCH --output=model_training_S_D_BN_C.%j.log
#SBATCH --error=model_training_S_D_BN_C.%j.log
#SBATCH --partition=gpu-a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

source "$WORK/conda/miniconda3/etc/profile.d/conda.sh"
conda activate model-training

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

echo "Starting Python script..."

python -u model_training.py \
  --hidden_sizes 512,512 \
  --dropout 0.5 \
  --use_batchnorm \
  --use_contrastive \
  --model_name face2voice_S_D_BN_C

echo "Job finished at $(date)"
