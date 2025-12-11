# ```
# conda init
# conda create -n embedding_preprocessing python=3.10 -y
# conda activate embedding_preprocessing
# conda install -c conda-forge pandas numpy tqdm -y
# pip install torch
# ```

import os
import random
import torch
import pandas as pd
from tqdm import tqdm

BASE = os.path.join(os.environ["SCRATCH"], "VoxCeleb2", "Embeddings")
SHARDS = [f"vox2_mp4_{i}" for i in range(1, 7)]



# LOAD AND CLEAN A SINGLE SHARD
def load_shard(shard_name):
    """Load one shard, remove zero vectors, align image/audio, return merged metadata + embeddings + removal stats."""

    shard_path = os.path.join(BASE, shard_name)
    k = shard_name.split("_")[-1]

    # Load metadata CSVs
    jpg_csv = os.path.join(shard_path, f"jpg_metadata_{k}.csv")
    wav_csv = os.path.join(shard_path, f"wav_metadata_{k}.csv")
    df_img = pd.read_csv(jpg_csv).rename(columns={"img_num": "segment"})
    df_wav = pd.read_csv(wav_csv).rename(columns={"audio_num": "segment"})

    # Load embedding tensors
    emb_img = torch.load(os.path.join(shard_path, f"jpg_embeddings_{k}.pt"))
    emb_wav = torch.load(os.path.join(shard_path, f"wav_embeddings_{k}.pt"))

    # Merge image/audio metadata first
    merged = pd.merge(
        df_img, df_wav,
        on=["id", "hash", "segment"],
        suffixes=("_img", "_wav")
    )

    # Keep original indices for sanity checks
    merged = merged.rename(columns={"index_img": "orig_index_img", "index_wav": "orig_index_wav"})

    # Remove zero embeddings
    img_mask = (emb_img[merged["orig_index_img"].values].abs().sum(dim=1) != 0)
    wav_mask = (emb_wav[merged["orig_index_wav"].values].abs().sum(dim=1) != 0)
    keep_mask = img_mask & wav_mask

    # Convert to NumPy for safe Pandas indexing
    keep_mask_np = keep_mask.cpu().numpy() if torch.is_tensor(keep_mask) else keep_mask

    # Count removed rows
    num_removed_img_zeros = (~img_mask).sum().item()
    num_removed_wav_zeros = (~wav_mask).sum().item()
    num_removed_unpaired = (~keep_mask).sum().item()

    # Apply mask 
    merged = merged.loc[keep_mask_np].reset_index(drop=True)

    # Align embeddings
    img_indices = merged["orig_index_img"].values
    wav_indices = merged["orig_index_wav"].values
    aligned_img_emb = emb_img[img_indices]
    aligned_wav_emb = emb_wav[wav_indices]

    # Return
    removal_stats = {
        "zeros_img": num_removed_img_zeros,
        "zeros_wav": num_removed_wav_zeros,
        "unpaired": num_removed_unpaired
    }

    return merged, aligned_img_emb, aligned_wav_emb, removal_stats



# PROCESS ALL SHARDS
all_metadata = []
all_img_embs = []
all_wav_embs = []

# Cumulative removal counters
total_removed_img_zeros = 0
total_removed_wav_zeros = 0
total_removed_unpaired = 0

for shard in tqdm(SHARDS, desc="Processing shards"):
    md, img, wav, stats = load_shard(shard)
    all_metadata.append(md)
    all_img_embs.append(img)
    all_wav_embs.append(wav)

    # Accumulate totals
    total_removed_img_zeros += stats["zeros_img"]
    total_removed_wav_zeros += stats["zeros_wav"]
    total_removed_unpaired += stats["unpaired"]

# Combine across shards
final_metadata = pd.concat(all_metadata).reset_index(drop=True)
final_img_embs = torch.cat(all_img_embs, dim=0)
final_wav_embs = torch.cat(all_wav_embs, dim=0)

# Add final index
final_metadata["index"] = range(len(final_metadata))

# Report total removed rows
print("==== TOTAL REMOVED ROWS ====")
print(f"Zero image rows: {total_removed_img_zeros}")
print(f"Zero audio rows: {total_removed_wav_zeros}")
print(f"Rows removed due to zeros/unpaired: {total_removed_unpaired}")



# SAVE FINAL DATA
OUT_DIR = os.path.join(BASE, "final")
os.makedirs(OUT_DIR, exist_ok=True)

final_metadata.to_csv(os.path.join(OUT_DIR, "final_metadata_train.csv"), index=False)
torch.save(final_img_embs, os.path.join(OUT_DIR, "final_image_embeddings_train.pt"))
torch.save(final_wav_embs, os.path.join(OUT_DIR, "final_audio_embeddings_train.pt"))



# SANITY CHECK FUNCTIONS
def load_original_from_metadata(row):
    """Reconstruct original embedding using only metadata (id/hash/segment)."""

    # Recover shard
    filepath = row["filepath_img"]
    shard_name = filepath.split("/")[-5]
    k = shard_name.split("_")[-1]
    shard_path = os.path.join(BASE, shard_name)

    # Load original metadata
    df_img_orig = pd.read_csv(os.path.join(shard_path, f"jpg_metadata_{k}.csv"))
    df_img_orig = df_img_orig.rename(columns={"img_num": "segment"})

    df_wav_orig = pd.read_csv(os.path.join(shard_path, f"wav_metadata_{k}.csv"))
    df_wav_orig = df_wav_orig.rename(columns={"audio_num": "segment"})

    # Find the row using id/hash/segment
    cond_img = (
        (df_img_orig["id"] == row["id"]) &
        (df_img_orig["hash"] == row["hash"]) &
        (df_img_orig["segment"] == row["segment"])
    )
    cond_wav = (
        (df_wav_orig["id"] == row["id"]) &
        (df_wav_orig["hash"] == row["hash"]) &
        (df_wav_orig["segment"] == row["segment"])
    )

    img_idx = df_img_orig.index[cond_img][0]
    wav_idx = df_wav_orig.index[cond_wav][0]

    # Load original embeddings
    img_emb_orig = torch.load(os.path.join(shard_path, f"jpg_embeddings_{k}.pt"))[img_idx]
    wav_emb_orig = torch.load(os.path.join(shard_path, f"wav_embeddings_{k}.pt"))[wav_idx]

    return img_emb_orig, wav_emb_orig


# SANITY CHECKS
print("\nRunning sanity checks...\n")

# 1. Length consistency
assert len(final_metadata) == final_img_embs.shape[0], "Mismatch: metadata vs image embeddings"
assert len(final_metadata) == final_wav_embs.shape[0], "Mismatch: metadata vs audio embeddings"
print("Final row counts match")

# 2. No zero vectors
assert (final_img_embs.abs().sum(dim=1) == 0).sum() == 0, "Zero image vectors found!"
assert (final_wav_embs.abs().sum(dim=1) == 0).sum() == 0, "Zero audio vectors found!"
print("No zero vectors remain")

# 3. Random metadata-based reconstruction tests
num_checks = 10
rows_to_check = random.sample(range(len(final_metadata)), num_checks)

for i in rows_to_check:
    row = final_metadata.iloc[i]
    img_orig, wav_orig = load_original_from_metadata(row)

    if not torch.allclose(final_img_embs[i], img_orig, atol=1e-6):
        raise AssertionError(f"Image mismatch at row {i}")

    if not torch.allclose(final_wav_embs[i], wav_orig, atol=1e-6):
        raise AssertionError(f"Audio mismatch at row {i}")

print("Random spot checks match original embeddings")

# 4. Ensure no duplicates
duplicates = final_metadata.duplicated(subset=["id", "hash", "segment"]).sum()
assert duplicates == 0, f"Found {duplicates} duplicate metadata entries!"
print("No duplicates detected")

print("\nAll sanity checks passed! Dataset is consistent and aligned.\n")
