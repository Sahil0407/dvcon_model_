"""
Task-Aware Dataset Loader for DVCon YOLOv8n (Batch-by-Task)
─────────────────────────────────────────────────────────────
Each image is assigned a SINGLE dominant task. Images are grouped by task
during batching so that all images in a batch share the same task embedding.

This ensures:
  1. FiLM conditioning is consistent within each batch
  2. The model learns: "given task X, detect objects relevant to X"
  3. When no relevant objects exist → model learns to output "no object found"

YOLO label format (each row in .txt):
    task_id  cx  cy  w  h   (all normalized 0-1)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms as T
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from configs.task_config import TASK_NAMES, PATHS
from utils.task_embeddings import load_task_embeddings


class TaskAwareCOCODataset(Dataset):
    """
    Loads filtered COCO images + YOLO-format labels.
    Returns (image_tensor, labels_tensor, task_id).

    Each image has exactly ONE task_id (the dominant task).
    Task embeddings are NOT stored per-sample — they are looked up by task_id
    in the collate function, ensuring batch-level consistency.
    """

    def __init__(
        self,
        img_dir:         str,
        lbl_dir:         str,
        embeddings_path: str  = PATHS["task_embeddings"],
        img_size:        int  = 640,
        augment:         bool = False,
    ):
        self.img_dir   = Path(img_dir)
        self.lbl_dir   = Path(lbl_dir)
        self.img_size  = img_size
        self.augment   = augment

        # Load precomputed task embeddings [14, 384]
        self.task_embs = load_task_embeddings(embeddings_path, device="cpu")

        # Pair images with their label files and extract task_id
        self.samples, self.task_to_indices = self._index_samples()

        # Transforms
        base = [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3,
                          saturation=0.2, hue=0.05),
        ] if augment else []

        self.transform = T.Compose(aug + base)

    def _index_samples(self):
        """
        Index all (image, label) pairs and group by task_id.
        Also scans for negative example label files (*_neg_t{N}.txt)
        which are empty label files signaling 'no object found' for that task.

        Returns:
            samples: list of (img_path, lbl_path, task_id)
            task_to_indices: dict mapping task_id → list of sample indices
        """
        samples = []
        task_to_indices = defaultdict(list)

        # ── Scan for negative example label files ──────────────────────────────
        # These have names like: {image_stem}_neg_t{task_id}.txt
        # They reference the same image but signal 'no object found' for that task.
        for lbl_path in sorted(self.lbl_dir.iterdir()):
            if not lbl_path.stem.endswith("_neg_t"):
                continue
            try:
                parts = lbl_path.stem.rsplit("_neg_t", 1)
                img_stem = parts[0]
                task_id  = int(parts[1])
            except (ValueError, IndexError):
                continue

            if not (0 <= task_id < len(TASK_NAMES)):
                continue

            # Find the corresponding image
            img_path = self._find_image(img_stem)
            if img_path is None:
                continue

            idx = len(samples)
            samples.append((img_path, lbl_path, task_id))
            task_to_indices[task_id].append(idx)

        # ── Scan for positive example label files ──────────────────────────────
        # Track which (image, label_path) pairs we've already indexed
        indexed_pairs = {(str(s[0]), str(s[1])) for s in samples}

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            # Skip if this exact (image, label) pair was already indexed
            # (e.g., as a negative example). But don't skip the image entirely
            # because it might have BOTH a positive and negative label.
            if (str(img_path), str(lbl_path)) in indexed_pairs:
                continue

            task_id = self._extract_task_id_from_label(lbl_path)
            if task_id is None:
                continue

            idx = len(samples)
            samples.append((img_path, lbl_path, task_id))
            task_to_indices[task_id].append(idx)

        return samples, dict(task_to_indices)

    def _find_image(self, stem: str):
        """Find an image file by stem name (without extension)."""
        for ext in [".jpg", ".jpeg", ".png"]:
            path = self.img_dir / (stem + ext)
            if path.exists():
                return path
        return None

    def _extract_task_id_from_label(self, lbl_path):
        """Extract the dominant task_id from a positive label file."""
        with open(lbl_path) as f:
            first_line = f.read().strip().splitlines()
            if first_line and first_line[0].strip():
                parts = first_line[0].strip().split()
                if parts:
                    return int(float(parts[0]))
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path, task_id = self.samples[idx]

        # ── Image ──────────────────────────────────────────────────────────────
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # [3, H, W]

        # ── Labels ─────────────────────────────────────────────────────────────
        labels = []
        with open(lbl_path) as f:
            for line in f.read().strip().splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(v) for v in parts])

        if labels:
            labels = torch.tensor(labels, dtype=torch.float32)  # [N, 5]
        else:
            # Empty label file (negative example) — no bounding boxes
            labels = torch.zeros(0, 5, dtype=torch.float32)

        return image, labels, task_id


class TaskGroupedBatchSampler(Sampler):
    """
    Custom sampler that groups images by task_id within each batch.

    Each batch contains images from ONE task. Tasks are cycled through
    in round-robin order. Within each task, images are shuffled.

    This ensures the FiLM conditioner receives a single, consistent
    task embedding for all images in the batch.
    """

    def __init__(self, task_to_indices, batch_size, drop_last=True):
        """
        Parameters
        ----------
        task_to_indices : dict[int, list[int]]
            Mapping from task_id to list of dataset indices.
        batch_size : int
            Number of images per batch.
        drop_last : bool
            Whether to drop the last incomplete batch.
        """
        self.task_to_indices = task_to_indices
        self.batch_size      = batch_size
        self.drop_last       = drop_last

        # Shuffle indices within each task
        for task_id in self.task_to_indices:
            random.shuffle(self.task_to_indices[task_id])

        # Build batches: round-robin through tasks
        self.batches = self._build_batches()

    def _build_batches(self):
        batches = []
        # Create iterators for each task
        task_iters = {
            tid: iter(indices)
            for tid, indices in self.task_to_indices.items()
        }
        task_ids = list(task_iters.keys())

        while task_ids:
            exhausted = []
            for tid in task_ids:
                # Try to fill batch_size samples from this task
                task_batch = []
                for _ in range(self.batch_size):
                    try:
                        task_batch.append(next(task_iters[tid]))
                    except StopIteration:
                        break
                if task_batch:
                    batches.append(task_batch)
                if len(task_batch) < self.batch_size:
                    exhausted.append(tid)
            # Remove exhausted tasks
            task_ids = [t for t in task_ids if t not in exhausted]

        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def collate_fn(batch):
    """
    Custom collate for batch-by-task loading.

    All images in the batch share the same task_id, so we look up
    the task embedding ONCE for the entire batch.

    Returns:
        images    : [B, 3, H, W]
        labels    : list of [N_i, 5] tensors  (variable length — YOLO expects this)
        task_embs : [B, 384]  — all rows are identical (same task)
        task_id   : int       — the task index for this batch
    """
    images, labels_list, task_ids = zip(*batch)

    images    = torch.stack(images, 0)
    task_id   = task_ids[0]  # All same in a batch

    return images, list(labels_list), task_id


def build_dataloaders(
    train_img_dir:    str,
    train_lbl_dir:    str,
    val_img_dir:      str,
    val_lbl_dir:      str,
    embeddings_path:  str,
    batch_size:       int = 16,
    num_workers:      int = 4,
    img_size:         int = 640,
):
    train_ds = TaskAwareCOCODataset(
        train_img_dir, train_lbl_dir,
        embeddings_path, img_size, augment=True
    )
    val_ds = TaskAwareCOCODataset(
        val_img_dir, val_lbl_dir,
        embeddings_path, img_size, augment=False
    )

    # Grouped sampler for training (batch-by-task)
    train_sampler = TaskGroupedBatchSampler(
        train_ds.task_to_indices,
        batch_size=batch_size,
        drop_last=True,
    )

    # For validation, use standard sequential batching
    # (but still grouped by task for consistent FiLM conditioning)
    val_sampler = TaskGroupedBatchSampler(
        val_ds.task_to_indices,
        batch_size=batch_size,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"📦 Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    print(f"   Train batches: {len(train_sampler)} | Val batches: {len(val_sampler)}")
    print(f"   Tasks with images: {len(train_ds.task_to_indices)}")

    return train_loader, val_loader
