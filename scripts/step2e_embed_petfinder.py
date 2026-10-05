"""
Step 2 (e): embed the PetFinder.my descriptions and first photos ONCE, with
the project's real adapters, so every later experiment runs on cached
vectors instead of re-running torch per draw.

Uses adapters.text.TextAdapter / adapters.image.ImageAdapter directly -- the
exact code path /fit and /analyze use -- so the validation measures the real
embeddings, not a lookalike.

Output (gitignored, lives next to the dataset):
  datasets/petfinder/_step2e_text_emb.npz   ids, emb (N,384)
  datasets/petfinder/_step2e_image_emb.npz  ids, emb (N,512)
  datasets/petfinder/_step2e_meta.csv       PetID, Type, Age, Breed1, Gender, AdoptionSpeed

Run (offline-safe; sandbox has no reliable HF access):
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/step2e_embed_petfinder.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from adapters.image import ImageAdapter  # noqa: E402
from adapters.text import TextAdapter  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "petfinder")
IMG_DIR = os.path.join(DATA, "train_images")
TEXT_OUT = os.path.join(DATA, "_step2e_text_emb.npz")
IMAGE_OUT = os.path.join(DATA, "_step2e_image_emb.npz")
META_OUT = os.path.join(DATA, "_step2e_meta.csv")
CHUNK = 64


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    df = pd.read_csv(os.path.join(DATA, "train.csv"),
                     usecols=["PetID", "Type", "Age", "Breed1", "Gender", "AdoptionSpeed", "Description"])
    df["Description"] = df["Description"].fillna("").astype(str)
    df.drop(columns=["Description"]).to_csv(META_OUT, index=False)

    # ---- text ----
    if not os.path.exists(TEXT_OUT):
        tdf = df[df["Description"].str.strip().str.len() > 0].reset_index(drop=True)
        log(f"embedding {len(tdf)} descriptions")
        adapter = TextAdapter()
        embs = []
        for i in range(0, len(tdf), CHUNK):
            embs.append(adapter.transform(tdf["Description"].iloc[i:i + CHUNK].tolist()))
            if (i // CHUNK) % 20 == 0:
                log(f"  text {i}/{len(tdf)}")
        np.savez(TEXT_OUT, ids=tdf["PetID"].to_numpy(), emb=np.vstack(embs))
        log("text done")
    else:
        log("text cache exists, skipping")

    # ---- image (first photo only; pets with no downloaded/readable photo are skipped, counted) ----
    if not os.path.exists(IMAGE_OUT):
        adapter = ImageAdapter()
        ids, embs, skipped = [], [], 0
        pet_ids = df["PetID"].tolist()
        log(f"embedding first photo for up to {len(pet_ids)} pets")
        for i in range(0, len(pet_ids), CHUNK):
            batch_ids, batch_raw = [], []
            for pid in pet_ids[i:i + CHUNK]:
                path = os.path.join(IMG_DIR, f"{pid}-1.jpg")
                if not os.path.exists(path):
                    skipped += 1
                    continue
                with open(path, "rb") as f:
                    batch_raw.append(f.read())
                batch_ids.append(pid)
            if not batch_raw:
                continue
            try:
                embs.append(adapter.transform(batch_raw))
                ids.extend(batch_ids)
            except Exception:
                # one undecodable file poisons a whole batch -- retry singly, count what fails
                for pid, raw in zip(batch_ids, batch_raw):
                    try:
                        embs.append(adapter.transform([raw]))
                        ids.append(pid)
                    except Exception:
                        skipped += 1
            if (i // CHUNK) % 10 == 0:
                log(f"  image {i}/{len(pet_ids)} (skipped so far: {skipped})")
        np.savez(IMAGE_OUT, ids=np.array(ids), emb=np.vstack(embs))
        log(f"image done: {len(ids)} embedded, {skipped} skipped (no photo / unreadable)")
    else:
        log("image cache exists, skipping")


if __name__ == "__main__":
    main()
