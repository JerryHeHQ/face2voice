#!/bin/bash
#SBATCH --job-name=extract_dataset
#SBATCH --output=extract_dataset.%j.log
#SBATCH --error=extract_dataset.%j.log
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"

# Load conda
source "$WORK/ls6/conda/miniconda3/etc/profile.d/conda.sh"
conda activate unzip-env

# Directory containing the downloaded zip files
SRC_DIR="$WORK/VoxCeleb2/Raw"

# Directory to extract to
DEST_DIR="$WORK/VoxCeleb2/Extracted"
mkdir -p "$DEST_DIR"

# Loop through all zip files in the source directory
for ZIP_FILE in "$SRC_DIR"/*.zip; do
    FILE_NAME=$(basename "$ZIP_FILE" .zip)
    EXTRACT_DIR="$DEST_DIR/$FILE_NAME"
    mkdir -p "$EXTRACT_DIR"
    echo "$(date +'%Y-%m-%d %H:%M:%S') - Extracting $ZIP_FILE to $EXTRACT_DIR..."
    7z x "$ZIP_FILE" -o"$EXTRACT_DIR" -mmt=16 -bb1
done

echo "Job finished at $(date)"

