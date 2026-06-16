"""
Task Embedding Generator for DVCon Task-Aware YOLOv8n
────────────────────────────────────────────────────────
Encodes the 14 task descriptions into fixed embedding vectors.
Run once → save to disk → reuse during training & inference.

Run:
    python utils/task_embeddings.py
"""

import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.task_config import TASK_NAMES, FILM_CONFIG, PATHS


def generate_task_embeddings(
    task_names:   list[str] = TASK_NAMES,
    model_name:   str       = FILM_CONFIG["text_encoder"],
    save_path:    str       = PATHS["task_embeddings"],
    verbose:      bool      = True,
) -> torch.Tensor:
    """
    Encode task descriptions → L2-normalized embedding matrix.

    Returns
    -------
    embeddings : torch.Tensor  shape [14, 384]
    """
    if verbose:
        print(f"Loading text encoder: {model_name}")

    encoder    = SentenceTransformer(model_name)
    embeddings = encoder.encode(
        task_names,
        convert_to_tensor  = True,
        normalize_embeddings = True,   # L2 normalize for cosine similarity
        show_progress_bar  = verbose,
    )                                  # shape: [14, 384]

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings.cpu(), save_path)

    if verbose:
        print(f"\n✅ Task embeddings saved → {save_path}")
        print(f"   Shape : {embeddings.shape}  (tasks × embed_dim)")
        print(f"   dtype : {embeddings.dtype}")
        print(f"\n   Cosine similarity matrix (should be ~diagonal):")
        sim = torch.mm(embeddings, embeddings.T).cpu().numpy()
        for i, name in enumerate(task_names):
            top2 = np.argsort(sim[i])[::-1][1]          # most similar ≠ self
            print(f"   [{i:2d}] {name:<30s}  → closest: [{top2}] {task_names[top2]}")

    return embeddings


def load_task_embeddings(
    path:   str    = PATHS["task_embeddings"],
    device: str    = "cpu",
) -> torch.Tensor:
    """Load precomputed embeddings from disk."""
    emb = torch.load(path, map_location=device)
    return emb    # [14, 384]


if __name__ == "__main__":
    generate_task_embeddings(verbose=True)
