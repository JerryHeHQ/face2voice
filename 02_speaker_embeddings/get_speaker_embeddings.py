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
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


# In[ ]:


# Paths
OPENVOICE_DIR = Path(os.path.expandvars("$SCRATCH/face2voice/OpenVoice"))
ROOT = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Extracted/vox2_mp4_1"))
CSV_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/wav_metadata_1.csv"))
EMB_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/wav_embeddings_1.pt"))
CHUNK_DIR = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/chunks"))
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
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


# Background thread for writing CSV & embeddings
executor = ThreadPoolExecutor(max_workers=8)
def async_save(chunk_id, embeddings, rows):
    """Save embeddings + CSV rows in a background thread."""
    torch.save(embeddings, CHUNK_DIR / f"emb_{chunk_id:06d}.pt")
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


# In[ ]:


# Main Loop
cache_embeddings = []
cache_rows = []
existing_chunks = sorted(CHUNK_DIR.glob("*.pt"))
chunk_id = len(existing_chunks)
next_index = start_index
skip_count = 0

for file_path in tqdm(wav_files[start_index:], desc="Extracting"):
    try:
        with torch.no_grad():
            se, _ = se_extractor.get_se(str(file_path), tone_color_converter, vad=True)
            emb = se.squeeze(-1).cpu() 
    except Exception as e:
        skip_count += 1
        print(f"[Error] Skipped {file_path}: {e}")
        continue

    # Store
    cache_embeddings.append(emb)
    parts = file_path.parts
    cache_rows.append([next_index, parts[-3], parts[-2], int(file_path.stem), str(file_path)])
    next_index += 1

    # Save chunk
    if len(cache_embeddings) >= SAVE_CHUNK:
        embeddings_tensor = torch.stack(cache_embeddings)

        # Save asynchronously
        executor.submit(async_save, chunk_id, embeddings_tensor, cache_rows)

        chunk_id += 1
        cache_embeddings = []
        cache_rows = []


# In[ ]:


# Save remaining embeddings & CSV rows
if len(cache_embeddings) > 0:
    embeddings_tensor = torch.stack(cache_embeddings)
    executor.submit(async_save, chunk_id, embeddings_tensor, cache_rows)
    
executor.shutdown(wait=True)
print(f"[Info] Total skipped embeddings: {skip_count}")


# In[ ]:


# Combine chunks into final embeddings file
chunks = []
for f in sorted(CHUNK_DIR.glob("*.pt")):
    chunks.append(torch.load(f))

all_embeddings = torch.cat(chunks, dim=0)
torch.save(all_embeddings, EMB_PATH)
print(f"Saved combined embeddings to {EMB_PATH}")


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

