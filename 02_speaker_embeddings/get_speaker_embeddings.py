#!/usr/bin/env python
# coding: utf-8

# Note that the openvoice conda environment should already have been set up according to the OpenVoice documentation.
# ```
# conda init
# conda activate openvoice
# pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# pip install pandas
# pip install ipykernel
# ```

# In[ ]:


import os
import sys
sys.path.append(os.path.expandvars("$SCRATCH/face2voice/OpenVoice"))


# In[ ]:


import csv
from pathlib import Path
import torch
import torchaudio
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


# In[ ]:


# Paths
OPENVOICE_DIR = Path(os.path.expandvars("$SCRATCH/face2voice/OpenVoice"))
ROOT = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Extracted/vox2_mp4_1"))
CSV_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/wav_metadata_1.csv"))
EMB_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/wav_embeddings_1.pt"))
SAVE_CHUNK = 512 


# In[ ]:


# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# In[ ]:


# Initialize OpenVoice ToneColorConverter
ckpt_converter = OPENVOICE_DIR / "checkpoints_v2/converter"
tone_color_converter = ToneColorConverter(str(ckpt_converter / "config.json"), device=device)
tone_color_converter.load_ckpt(str(ckpt_converter / "checkpoint.pth"))


# In[ ]:


# Resume state logic
def load_resume_state():
    """Reads last CSV index to resume correctly."""
    if not CSV_PATH.exists():
        return 0

    with open(CSV_PATH, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

        if len(rows) <= 1:
            return 0

        last_index = int(rows[-1][0])
        print(f"[Resume] Last completed index: {last_index}")
        return last_index + 1


# In[ ]:


# Ensure CSV exists with header
if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "id", "hash", "audio_num", "filepath"])

start_index = load_resume_state()


# In[ ]:


# Scan filesystem
wav_files = sorted(ROOT.rglob("*.wav"))
total = len(wav_files)
print(f"Found {total} wav files.")


# In[ ]:


# Load existing embeddings in case resuming
if EMB_PATH.exists():
    all_embeddings = torch.load(EMB_PATH)
    print(f"[Resume] Loaded existing embedding tensor: {all_embeddings.shape}")
else:
    all_embeddings = torch.zeros((0, 256), dtype=torch.float32)


# In[ ]:


# Helper function
def process_file(idx, file_path):
    """Process a single WAV file: extract embedding and build CSV row."""
    try:
        with torch.no_grad():
            se, _ = se_extractor.get_se(str(file_path), tone_color_converter, vad=True)
            emb = se.squeeze(-1)
        parts = file_path.parts
        row = [idx, parts[-3], parts[-2], int(file_path.stem), str(file_path)]
        return idx, emb, row, None
    except Exception as e:
        return idx, None, None, e


# In[ ]:


# Main loop
cache_embeddings = []
cache_rows = []
next_index = start_index
skip_count = 0

MAX_THREADS = 16
with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    # Submit all files
    futures = [executor.submit(process_file, idx, fp) for idx, fp in enumerate(wav_files[start_index:], start_index)]

    # Process results in order of submission to preserve wav_files order
    for future in futures:
        idx, emb, row, err = future.result()
        if err:
            skip_count += 1
            print(f"[Error] Skipped {idx}: {err}")
            continue

        cache_embeddings.append(emb)
        cache_rows.append(row)
        next_index += 1

        # Save in batches
        if len(cache_embeddings) >= SAVE_CHUNK:
            # Concatenate embeddings and move to CPU
            chunk_tensor = torch.cat(cache_embeddings, dim=0).cpu()
            all_embeddings = torch.cat([all_embeddings, chunk_tensor], dim=0)
            torch.save(all_embeddings, EMB_PATH)
            cache_embeddings = []

            # Write CSV rows in batch
            with open(CSV_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(cache_rows)
            cache_rows = []


# In[ ]:


# Save remaining embeddings & CSV rows
if len(cache_embeddings) > 0:
    chunk_tensor = torch.cat(cache_embeddings, dim=0).cpu()
    all_embeddings = torch.cat([all_embeddings, chunk_tensor], dim=0)
    torch.save(all_embeddings, EMB_PATH)

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cache_rows)

print(f"[Info] Total skipped embeddings: {skip_count}")


# In[ ]:


# Check CSV File
df = pd.read_csv(CSV_PATH)
print(len(df))
df.head()


# In[ ]:


# Check Embeddings File
embeddings = torch.load(EMB_PATH)
print(embeddings.shape)
print("Has zeros:", (embeddings.abs().sum(dim=1) == 0).any())
print(embeddings[0][:6])


# In[ ]:


# Check Embeddings
index = 67

row = df.iloc[index]
print(row)

vec = embeddings[index]

print("Norm:", torch.norm(vec).item())
print("Min value:", vec.min().item())
print("Max value:", vec.max().item())
print("Embedding vector:", vec[:6])

