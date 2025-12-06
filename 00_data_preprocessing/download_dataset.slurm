#!/bin/bash
#SBATCH --job-name=download_dataset
#SBATCH --output=download_dataset.%j.log
#SBATCH --error=download_dataset.%j.log
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=16G
#SBATCH --time=08:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"

# Directory to save downloads
DEST_DIR="$SCRATCH/VoxCeleb2/Raw"
mkdir -p "$DEST_DIR"

# List of file URLs
FILES=(
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_1.zip?download=true"
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_2.zip?download=true"
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_3.zip?download=true"
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_4.zip?download=true"
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_5.zip?download=true"
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_mp4_6.zip?download=true"
    "https://huggingface.co/datasets/ProgramComputer/voxceleb/resolve/main/vox2/vox2_test_mp4.zip?download=true"
)

# Download files in parallel
printf "%s\n" "${FILES[@]}" | xargs -n 1 -P 7 -I {} bash -c '
FILE_NAME=$(basename "{}" | cut -d"?" -f1)
LOG_FILE="'"$DEST_DIR"'/$FILE_NAME.log"
echo "$(date +\"%H:%M:%S\") - Downloading $FILE_NAME..."
wget -c --content-disposition "{}" -P "'"$DEST_DIR"'" --tries=5 --timeout=30 &> "$LOG_FILE"
'

echo "Job finished at $(date)"
