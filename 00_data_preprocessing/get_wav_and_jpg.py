#!/usr/bin/env python
# coding: utf-8

# ```
# conda init
# conda create -n get-wav-and-jpg python=3.10 -y
# conda activate get-wav-and-jpg
# conda install -c conda-forge ffmpeg
# pip install ipykernel
# ```

# In[ ]:


import os
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


# In[ ]:


# Paths
ROOT = os.path.expandvars("$SCRATCH/VoxCeleb2/Extracted/vox2_mp4_test")
MP4_DIR = Path(ROOT) / "mp4"
WAV_DIR = Path(ROOT) / "wav"
JPG_DIR = Path(ROOT) / "jpg"


# In[ ]:


def ensure_dir(path: Path):
    """Ensure the directory exists."""
    path.mkdir(parents=True, exist_ok=True)


# In[ ]:


def process_file(mp4_path: Path, wav_path: Path, jpg_path: Path):
    """Extract WAV audio and JPG frame from a single MP4."""
    ensure_dir(wav_path.parent)
    ensure_dir(jpg_path.parent)

    if wav_path.exists() and jpg_path.exists():
        return

    # ffmpeg command to extract audio and first video frame
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp4_path),
        "-map", "0:a", "-ac", "1", "-ar", "22050", "-acodec", "pcm_s16le", str(wav_path),
        "-map", "0:v:0", "-vframes", "1", str(jpg_path)
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        print(f"ERROR processing {mp4_path}:\n{e.stderr.decode()}")


# In[ ]:


def process_vox2(speaker_id: str, max_workers=8):
    """Process all MP4s for a given speaker in parallel."""
    search_dir = MP4_DIR / speaker_id

    # Check speaker folder
    if not search_dir.exists():
        print(f"Speaker folder does not exist: {search_dir}")
        return

    # Find all MP4s
    mp4_files = list(search_dir.rglob("*.mp4"))
    total = len(mp4_files)
    if total == 0:
        print(f"No MP4 files found in {search_dir}")
        return

    tasks = []
    for mp4_path in mp4_files:
        relative = mp4_path.relative_to(MP4_DIR)
        wav_out = WAV_DIR / relative.with_suffix(".wav")
        jpg_out = JPG_DIR / relative.with_suffix(".jpg")
        tasks.append((mp4_path, wav_out, jpg_out))

    print(f"[{speaker_id}] Processing {total} files with {max_workers} workers...")

    # Run extractions in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_file, *task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), start=1):
            if i % 50 == 0 or i == 1:
                print(f"[{speaker_id}] Completed {i}/{total}")

    print(f"DONE processing speaker {speaker_id}.")


# In[ ]:


# List all speaker directories
speaker_ids = [
    name for name in os.listdir(MP4_DIR)
    if os.path.isdir(os.path.join(MP4_DIR, name))
]
print(len(speaker_ids), "speakers found.")


# In[ ]:


# Process each speaker sequentially
for i, speaker_id in enumerate(speaker_ids):
    print(f"Processing speaker {i}/{len(speaker_ids)}: {speaker_id}")
    process_vox2(speaker_id, max_workers=64)

