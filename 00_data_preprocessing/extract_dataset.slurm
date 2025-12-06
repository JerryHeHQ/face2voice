#!/bin/bash
#SBATCH --job-name=extract_dataset
#SBATCH --output=extract_dataset.%j.log
#SBATCH --error=extract_dataset.%j.log
#SBATCH --partition=development
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=64G
#SBATCH --time=02:00:00

echo "Job started at $(date)"
echo "Running on node $(hostname)"

# Load conda
source "$WORK/conda/miniconda3/etc/profile.d/conda.sh"
conda activate unzip-env

# Directory containing the downloaded zip files
SRC_DIR="$WORK/VoxCeleb2/Raw"

# Directory to extract to
DEST_DIR="$WORK/VoxCeleb2/Extracted"
mkdir -p "$DEST_DIR"

# Number of parallel extractions
PARALLEL_JOBS=7
CPUS_PER_JOB=$(( SLURM_CPUS_PER_TASK / PARALLEL_JOBS ))

echo "Running up to $PARALLEL_JOBS extractions in parallel, $CPUS_PER_JOB threads per extraction"

# Export variables so xargs/batch processes can see them
export DEST_DIR

# Find all zip files and run extraction in parallel
find "$SRC_DIR" -maxdepth 1 -name '*.zip' | xargs -n 1 -P $PARALLEL_JOBS -I {} bash -c "
ZIP_FILE='{}'
FILE_NAME=\$(basename \"\$ZIP_FILE\" .zip)
EXTRACT_DIR=\"$DEST_DIR/\$FILE_NAME\"
mkdir -p \"\$EXTRACT_DIR\"
LOG_FILE=\"\$EXTRACT_DIR/extraction.log\"

echo '-----------------------------------------------------'
echo \"\$(date +'%Y-%m-%d %H:%M:%S') - Starting extraction of \$ZIP_FILE\"
echo \"Extraction log: \$LOG_FILE\"
echo '-----------------------------------------------------'

if 7z x \"\$ZIP_FILE\" -o\"\$EXTRACT_DIR\" -mmt=$CPUS_PER_JOB -bb1 &> \"\$LOG_FILE\"; then
    echo \"\$(date +'%Y-%m-%d %H:%M:%S') - Successfully extracted \$ZIP_FILE\"
else
    echo \"\$(date +'%Y-%m-%d %H:%M:%S') - ERROR extracting \$ZIP_FILE. See \$LOG_FILE for details\"
fi

echo '-----------------------------------------------------'
"

echo "Job finished at $(date)"
