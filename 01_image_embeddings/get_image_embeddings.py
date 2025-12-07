#!/usr/bin/env python
# coding: utf-8

# ```
# conda init
# conda create -n get-image-embeddings python=3.10 -c conda -y
# conda activate get-image-embeddings
# pip install torch==2.2.1+cu118 torchvision==0.17.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
# pip install facenet-pytorch==2.6.0
# pip install pandas
# pip install ipykernel
# pip install tqdm
# ```

# In[ ]:


import os
import csv
from pathlib import Path
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
import pandas as pd
from IPython.display import display


# In[ ]:


# Paths
ROOT = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Extracted/vox2_mp4_1"))
CSV_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/jpg_metadata_test.csv"))
EMB_PATH = Path(os.path.expandvars("$SCRATCH/VoxCeleb2/Embeddings/vox2_mp4_1/jpg_embeddings_test.pt"))
SAVE_CHUNK = 512
BATCH_SIZE = 512


# In[3]:


# Device Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# In[4]:


# MTCNN | Face Detection & Alignment | Not Used For Now
mtcnn = MTCNN(
    image_size=160, 
    margin=0, 
    min_face_size=20, 
    device=device
    )

# Backup | Face Detection & Alignment | Currently Used (Assumes Dataset is Aligned)
fallback_preprocess = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# FaceNet | Face Embedding Model
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
for param in facenet.parameters():
    param.requires_grad = False


# In[5]:


def load_and_preprocess_image(image_path):
    return fallback_preprocess(Image.open(image_path).convert("RGB"))


# In[6]:


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


# In[7]:


# Ensure CSV exists with header
if not CSV_PATH.exists():
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "id", "hash", "img_num", "filepath"])

start_index = load_resume_state()


# In[8]:


# Scan filesystem
jpg_files = sorted(ROOT.rglob("*.jpg"))
total = len(jpg_files)
print(f"Found {total} jpg files.")


# In[9]:


# Load existing embeddings in case resuming
if EMB_PATH.exists():
    all_embeddings = torch.load(EMB_PATH)
    print(f"[Resume] Loaded existing embedding tensor: {all_embeddings.shape}")
else:
    all_embeddings = torch.zeros((0, 512), dtype=torch.float32)


# In[ ]:


cache_embeddings = []
next_index = start_index

for i in tqdm(range(start_index, total, BATCH_SIZE), desc="Processing JPGs"):
    batch_files = jpg_files[i:i+BATCH_SIZE]
    batch_imgs = [load_and_preprocess_image(f) for f in batch_files]
    faces_batch = torch.stack(batch_imgs).to(device)

    # Compute embeddings in full precision
    with torch.no_grad():
        emb = facenet(faces_batch)
        emb = torch.nn.functional.normalize(emb, dim=1)
    emb = emb.cpu()
    cache_embeddings.append(emb)

    # Build a list of rows for the batch
    batch_rows = []
    for idx, file_path in enumerate(batch_files):
        parts = file_path.parts
        batch_rows.append([next_index, parts[-3], parts[-2], int(Path(parts[-1]).stem), str(file_path)])
        next_index += 1

    # Write all rows at once
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(batch_rows)

    # Periodically save embeddings
    if len(cache_embeddings) * BATCH_SIZE >= SAVE_CHUNK:
        cache_embeddings = torch.cat(cache_embeddings, dim=0)
        all_embeddings = torch.cat([all_embeddings, cache_embeddings], dim=0)
        torch.save(all_embeddings, EMB_PATH)
        cache_embeddings = []


# In[ ]:


# Save remaining embeddings
if len(cache_embeddings) > 0:
    cache_embeddings = torch.cat(cache_embeddings, dim=0)
    all_embeddings = torch.cat([all_embeddings, cache_embeddings], dim=0)
    torch.save(all_embeddings, EMB_PATH)


# In[12]:


# Check CSV File
df = pd.read_csv(CSV_PATH)
print(len(df))
df.head()


# In[13]:


# Check Embeddings File
embeddings = torch.load(EMB_PATH)
print(embeddings.shape)
print("Has zeros:", (embeddings.abs().sum(dim=1) == 0).any())
print(embeddings[0][:6])


# In[14]:


# Check Embeddings
index = 67

row = df.iloc[index]
print(row)

vec = embeddings[index]

print("Norm:", torch.norm(vec).item())
print("Min value:", vec.min().item())
print("Max value:", vec.max().item())
print("Embedding vector:", vec[:6])


# In[15]:


# Display the image
img = Image.open(row["filepath"])
display(img)

