# ----------------------------------------------------
# ENVIRONMENT SETUP
# ----------------------------------------------------
# conda init
# conda create -n model-training python=3.10 -y
# conda activate model-training
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install pandas scikit-learn tqdm

import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random

# ----------------------------------------------------
# CLI ARGUMENTS
# ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--hidden_sizes", type=str, default="1024,1024",
                    help="Comma-separated hidden layer sizes, e.g. 1024,1024,512")
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--use_batchnorm", action="store_true", help="Include BatchNorm layers")
parser.add_argument("--use_contrastive", action="store_true", help="Use contrastive loss")
parser.add_argument("--model_name", type=str, default="mlp_final",
                    help="Prefix for saving the output model files")
args = parser.parse_args()

HIDDEN_SIZES = [int(x) for x in args.hidden_sizes.split(",")]
DROPOUT = args.dropout
USE_BATCHNORM = args.use_batchnorm
USE_CONTRASTIVE = args.use_contrastive
MODEL_NAME = args.model_name

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("\n===== TRAINING CONFIGURATION =====")
print(f"Hidden layer sizes: {HIDDEN_SIZES}")
print(f"Dropout: {DROPOUT}")
print(f"Use BatchNorm: {USE_BATCHNORM}")
print(f"Use Contrastive Loss: {USE_CONTRASTIVE}")
print(f"Model name: {MODEL_NAME}")
print(f"Device: {DEVICE}")
print("==================================\n")

# ----------------------------------------------------
# GLOBAL CONFIG
# ----------------------------------------------------
BASE = "/scratch/11077/jerryhe/VoxCeleb2/Embeddings/final"
METADATA_CSV = os.path.join(BASE, "final_metadata_train.csv")
IMG_EMB_PTH = os.path.join(BASE, "final_image_embeddings_train.pt")
WAV_EMB_PTH = os.path.join(BASE, "final_audio_embeddings_train.pt")

BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 5

LR = 5e-4
WEIGHT_DECAY = 1e-4
MARGIN = 0.1

# ----------------------------------------------------
# SEED EVERYTHING
# ----------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
metadata = pd.read_csv(METADATA_CSV)
img_embs = torch.load(IMG_EMB_PTH).to(torch.float32)
wav_embs = torch.load(WAV_EMB_PTH).to(torch.float32)

assert len(metadata) == img_embs.shape[0] == wav_embs.shape[0]

# ----------------------------------------------------
# DATASET CLASSES
# ----------------------------------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ContrastiveDataset(Dataset):
    def __init__(self, x, y, ids):
        self.x = x
        self.y = y
        self.ids = torch.tensor(ids, dtype=torch.long)

        self.id_to_indices = defaultdict(list)
        print("Building speaker index mapping...")
        for idx, spk in tqdm(enumerate(self.ids), total=len(self.ids)):
            self.id_to_indices[int(spk)].append(idx)

        self.length = len(self.ids) * 2  # pos + neg per sample

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample_idx = idx // 2
        label_type = idx % 2

        speaker = int(self.ids[sample_idx])

        if label_type == 0:
            # positive pair
            candidates = self.id_to_indices[speaker]
            if len(candidates) < 2:
                j = sample_idx
            else:
                j = sample_idx
                while j == sample_idx:
                    j = random.choice(candidates)
            label = 1

        else:
            # negative pair
            neg_speaker = speaker
            while neg_speaker == speaker:
                neg_speaker = int(self.ids[random.randint(0, len(self.ids) - 1)])
            candidates = self.id_to_indices[neg_speaker]
            j = random.choice(candidates)
            label = 0

        return self.x[sample_idx], self.y[j], torch.tensor(label, dtype=torch.float32)

# ----------------------------------------------------
# MODEL
# ----------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes, dropout, use_batchnorm):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------
# CONTRASTIVE LOSS
# ----------------------------------------------------
# def contrastive_loss(pred, target, label, margin):
#     cos = nn.functional.cosine_similarity(pred, target)
#     pos_loss = label * (1 - cos)
#     neg_loss = (1 - label) * torch.clamp(cos - margin, min=0)
#     return (pos_loss + neg_loss).mean()
def contrastive_loss(pred, target, label, margin):
    mse = nn.functional.mse_loss(pred, target, reduction="none").mean(dim=1)
    pos_loss = label * mse
    neg_loss = (1 - label) * torch.clamp(margin - mse, min=0)
    return (pos_loss + neg_loss).mean()

# ----------------------------------------------------
# K-FOLD CROSS VALIDATION
# ----------------------------------------------------
mse_loss = nn.MSELoss()
unique_ids = metadata["id"].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs_needed = []

for fold, (train_idx, val_idx) in enumerate(kf.split(unique_ids)):
    print(f"\n=== Fold {fold+1}/5 ===")

    train_ids = set(unique_ids[train_idx])
    val_ids = set(unique_ids[val_idx])

    train_mask = metadata["id"].isin(train_ids)
    val_mask = metadata["id"].isin(val_ids)

    img_train, wav_train = img_embs[train_mask], wav_embs[train_mask]
    img_val, wav_val = img_embs[val_mask], wav_embs[val_mask]

    ids_train_encoded, _ = pd.factorize(metadata.loc[train_mask, "id"])

    if USE_CONTRASTIVE:
        train_ds = ContrastiveDataset(img_train, wav_train, ids_train_encoded)
    else:
        train_ds = EmbeddingDataset(img_train, wav_train)

    val_ds = EmbeddingDataset(img_val, wav_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = MLP(img_embs.shape[1], wav_embs.shape[1], HIDDEN_SIZES, DROPOUT, USE_BATCHNORM).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    cosine = nn.CosineSimilarity(dim=1)

    best_val = float("inf")
    patience_counter = 0
    best_epoch = 1

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_sum = 0

        # Train
        for batch in tqdm(train_loader, leave=False, desc=f"Fold {fold+1} Train"):
            optimizer.zero_grad()

            if USE_CONTRASTIVE:
                x1, x2, labels = batch
                x1, x2, labels = x1.to(DEVICE), x2.to(DEVICE), labels.to(DEVICE)
                pred = model(x1)
                loss = contrastive_loss(pred, x2, labels, MARGIN)
                batch_size = x1.size(0)
            else:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                # loss = (1 - cosine(pred, y)).mean()
                loss = mse_loss(pred, y)
                batch_size = x.size(0)

            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_size

        # Validation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                # loss = (1 - cosine(pred, y)).mean()
                loss = mse_loss(pred, y)
                val_loss_sum += loss.item() * x.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        print(f"Fold {fold+1} Epoch {epoch}: Val Loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break

    epochs_needed.append(best_epoch)

# ----------------------------------------------------
# FINAL EPOCH COUNT
# ----------------------------------------------------
avg_epochs = max(1, int(np.mean(epochs_needed)))
print("\nEstimated epochs from CV:", avg_epochs)

# ----------------------------------------------------
# FINAL TRAINING ON FULL DATA
# ----------------------------------------------------
ids_encoded, _ = pd.factorize(metadata["id"])

if USE_CONTRASTIVE:
    train_ds_full = ContrastiveDataset(img_embs, wav_embs, ids_encoded)
else:
    train_ds_full = EmbeddingDataset(img_embs, wav_embs)

train_loader_full = DataLoader(train_ds_full, batch_size=BATCH_SIZE, shuffle=True)

model_full = MLP(img_embs.shape[1], wav_embs.shape[1], HIDDEN_SIZES, DROPOUT, USE_BATCHNORM).to(DEVICE)
optimizer = torch.optim.AdamW(model_full.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
cosine = nn.CosineSimilarity(dim=1)

print(f"\nTraining final model for {avg_epochs} epochs...\n")

for epoch in range(1, avg_epochs + 1):
    model_full.train()
    loss_sum = 0

    for batch in tqdm(train_loader_full, leave=False, desc=f"Final Train {epoch}/{avg_epochs}"):
        optimizer.zero_grad()

        if USE_CONTRASTIVE:
            x1, x2, labels = batch
            x1, x2, labels = x1.to(DEVICE), x2.to(DEVICE), labels.to(DEVICE)
            pred = model_full(x1)
            loss = contrastive_loss(pred, x2, labels, MARGIN)
            batch_size = x1.size(0)
        else:
            x, y = batch
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model_full(x)
            # loss = (1 - cosine(pred, y)).mean()
            loss = mse_loss(pred, y)
            batch_size = x.size(0)

        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch_size

    print(f"Epoch {epoch}: Train Loss = {loss_sum / len(train_loader_full.dataset):.6f}")

# ----------------------------------------------------
# SAVE MODELS
# ----------------------------------------------------
torch.save(model_full.state_dict(), os.path.join(BASE, f"{MODEL_NAME}.pt"))

model_full.eval()
scripted = torch.jit.script(model_full)
torchscript_path = os.path.join(BASE, f"{MODEL_NAME}_torchscript.pt")
scripted.save(torchscript_path)

print(f"Saved model: {MODEL_NAME}.pt")
print(f"Saved TorchScript model: {MODEL_NAME}_torchscript.pt")

# ----------------------------------------------------
# LOAD TEST DATA
# ----------------------------------------------------
metadata_test = pd.read_csv(os.path.join(BASE, "final_metadata_test.csv"))
img_test = torch.load(os.path.join(BASE, "final_image_embeddings_test.pt")).to(torch.float32)
wav_test = torch.load(os.path.join(BASE, "final_audio_embeddings_test.pt")).to(torch.float32)

test_loader = DataLoader(EmbeddingDataset(img_test, wav_test), batch_size=BATCH_SIZE)

# ----------------------------------------------------
# TEST EVALUATION
# ----------------------------------------------------
model_ts = torch.jit.load(torchscript_path, map_location=DEVICE)
model_ts.eval()

test_loss = 0.0
cos = nn.CosineSimilarity(dim=1)

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="Testing"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model_ts(x)
        # loss = (1 - cos(pred, y)).mean()
        loss = mse_loss(pred, y)
        test_loss += loss.item() * x.size(0)

test_loss /= len(test_loader.dataset)
print(f"Test set MSE loss: {test_loss:.6f}")
