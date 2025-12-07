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


# In[ ]:


# Paths & Configuration
OPENVOICE_DIR = Path(os.path.expandvars("$SCRATCH/face2voice/OpenVoice"))
ROOT = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Extracted/vox2_mp4_1"))
CSV_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/wav_metadata_1.csv"))
EMB_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/wav_embeddings_1.pt"))
SAVE_CHUNK = 1024


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


# Scan filesystem
wav_files = sorted(ROOT.rglob("*.wav"))
total = len(wav_files)
print(f"Found {total} wav files.")


# In[ ]:


# Pre-generate full CSV with all metadata
if not CSV_PATH.exists():
    print("[Init] Generating full metadata CSV...")
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "id", "hash", "audio_num", "filepath"])

        for idx, file_path in enumerate(tqdm(wav_files, desc="Generating CSV")):
            parts = file_path.parts
            row = [idx, parts[-3], parts[-2], int(Path(parts[-1]).stem), str(file_path)]
            writer.writerow(row)


# In[ ]:


# Load CSV
csv_df = pd.read_csv(CSV_PATH)
assert len(csv_df) == total
print("[Init] CSV loaded with correct row count.")


# In[ ]:


# Load existing embeddings to resume
if EMB_PATH.exists():
    all_embeddings_list = [emb for emb in torch.load(EMB_PATH)]
    print(f"[Resume] Loaded existing embeddings: {len(all_embeddings_list)}")
else:
    all_embeddings_list = []


# In[ ]:


# Resume based on embedding count only
start_index = len(all_embeddings_list)
print(f"[Resume] Resuming from embedding index: {start_index}")


# In[ ]:


# Main loop
skip_count = 0

for idx in tqdm(range(start_index, total), desc="Processing WAVs"):
    file_path = wav_files[idx]

    try:
        with torch.no_grad():
            se, _ = se_extractor.get_se(str(file_path), tone_color_converter, vad=True)
            se = se.float()
        emb = se.cpu().squeeze()
    except Exception as e:
        print(f"[Error] Failed extracting embedding for {file_path}: {e}")
        skip_count += 1
        emb = torch.zeros(256, dtype=torch.float32)
        print(f"[Info] Inserted zero vector. Total zeroed: {skip_count}")

    all_embeddings_list.append(emb)

    if len(all_embeddings_list) % SAVE_CHUNK == 0:
        torch.save(torch.stack(all_embeddings_list), EMB_PATH)


# In[ ]:


# Save remaining embeddings
torch.save(torch.stack(all_embeddings_list), EMB_PATH)
print(f"[Info] Total zero-vector fallbacks: {skip_count}")


# In[ ]:


# Check CSV File
df = pd.read_csv(CSV_PATH)
print(len(df))
df.head()


# In[ ]:


# Check Embeddings File
embeddings = torch.load(EMB_PATH)
print(embeddings.shape)
zero_vector_count = (embeddings.abs().sum(dim=1) == 0).sum().item()
print(f"Number of zero vectors: {zero_vector_count}")
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

