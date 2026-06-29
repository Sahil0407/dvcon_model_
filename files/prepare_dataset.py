"""
Dataset Preparation Script for DVCon Task-Aware YOLOv8n
─────────────────────────────────────────────────────────
1. Filters COCO 2017 images by task-relevant categories
2. Assigns ONE dominant task per image (most frequent task-relevant object)
3. Generates negative examples: images where task-relevant objects are absent
4. Converts COCO JSON annotations → YOLO .txt format
5. Creates dataset.yaml for ultralytics

Design: Batch-by-Task
────────────────────
Each image is assigned a single primary task. During training, images are
grouped by task so each batch contains images from the same task. This means:
  - The FiLM conditioner receives ONE task embedding per batch
  - The model learns: "given task X, detect objects relevant to X"
  - When no relevant objects exist → model learns to output "no object found"

Run:
    python prepare_dataset.py
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml
from tqdm import tqdm

import sys
from configs.task_config import (
    TASK_NAMES, TASK_TO_COCO_CATS, PATHS,
    TASKS_NEEDING_CUSTOM_DATA,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def coco_bbox_to_yolo(bbox, img_w, img_h):
    """Convert COCO [x,y,w,h] → YOLO [cx,cy,w,h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def build_cat_to_tasks(task_to_cats):
    """Reverse map: coco_cat_id → list of task_ids that need it."""
    cat_to_tasks = defaultdict(list)
    for task_id, cat_ids in task_to_cats.items():
        for cat_id in cat_ids:
            cat_to_tasks[cat_id].append(task_id)
    return cat_to_tasks


def assign_dominant_task(annotations, cat_to_tasks):
    """
    From all annotations on an image, pick the single dominant task.
    The dominant task is the one whose task-specific objects are most frequent.

    Returns (dominant_task_id, filtered_annotations) or (None, []) if no task
    can be assigned.
    """
    task_counts = defaultdict(int)
    task_anns   = defaultdict(list)

    for ann in annotations:
        cat_id = ann["category_id"]
        tasks  = cat_to_tasks.get(cat_id, [])
        for t in tasks:
            task_counts[t] += 1
            task_anns[t].append(ann)

    if not task_counts:
        return None, []

    # Pick the task with the most relevant objects
    dominant_task = max(task_counts.items(), key=lambda item: item[1])[0]
    # Return only annotations that belong to the dominant task
    dominant_cat_ids = set(TASK_TO_COCO_CATS[dominant_task])
    dominant_anns = [a for a in annotations if a["category_id"] in dominant_cat_ids]

    return dominant_task, dominant_anns


# ── Core Filter ────────────────────────────────────────────────────────────────

def filter_and_convert(ann_json_path, img_src_dir, out_img_dir, out_lbl_dir,
                       split="train"):
    """
    Filter COCO annotations → YOLO format with single-task-per-image labels.

    One label file per image. Each line = task_id cx cy w h
    All boxes in an image share the same task_id (the dominant task).

    Returns: dict mapping task_id → list of (img_filename) for negative sampling.
    """
    print(f"\n[{split}] Loading COCO annotations from {ann_json_path} ...")
    with open(ann_json_path) as f:
        coco = json.load(f)

    # Build lookup structures
    img_id_to_info = {img["id"]: img for img in coco["images"]}
    cat_to_tasks   = build_cat_to_tasks(TASK_TO_COCO_CATS)

    # Group annotations by image
    img_to_anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        if ann["category_id"] in cat_to_tasks:
            img_to_anns[ann["image_id"]].append(ann)

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    skipped   = 0
    processed = 0
    task_to_images = defaultdict(list)  # task_id → list of image filenames

    for img_id, anns in tqdm(img_to_anns.items(), desc=f"Processing {split}"):
        img_info = img_id_to_info[img_id]
        src_path = Path(img_src_dir) / img_info["file_name"]

        if not src_path.exists():
            skipped += 1
            continue

        # Assign single dominant task
        dominant_task, dominant_anns = assign_dominant_task(anns, cat_to_tasks)
        if dominant_task is None:
            continue

        img_w = img_info["width"]
        img_h = img_info["height"]

        label_lines = []
        for ann in dominant_anns:
            cx, cy, nw, nh = coco_bbox_to_yolo(ann["bbox"], img_w, img_h)
            # Clamp to [0,1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))
            label_lines.append(f"{dominant_task} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not label_lines:
            continue

        # Copy image
        dst_img = Path(out_img_dir) / img_info["file_name"]
        shutil.copy2(src_path, dst_img)

        # Write label file
        stem     = Path(img_info["file_name"]).stem
        lbl_path = Path(out_lbl_dir) / f"{stem}.txt"
        with open(lbl_path, "w") as lf:
            lf.write("\n".join(label_lines))

        task_to_images[dominant_task].append(img_info["file_name"])
        processed += 1

    print(f"  ✅ Processed: {processed} images")
    print(f"  ⚠️  Skipped (missing):  {skipped} images")

    # Print per-task distribution
    print(f"\n  Per-task image counts ({split}):")
    for tid in sorted(task_to_images.keys()):
        print(f"    Task {tid:2d} ({TASK_NAMES[tid][:25]:<25s}): {len(task_to_images[tid]):5d} images")

    return task_to_images


def generate_negative_examples(all_task_images,
                               out_lbl_dir,
                               max_negatives_per_task=200):
    """
    For each task, collect images that do NOT contain task-relevant objects.
    These serve as negative examples teaching the model "no object found".

    Each negative image gets a label file with ONLY the task_id header
    and NO bounding boxes — this signals to the model that nothing relevant
    is present for this task.
    """
    print("\n🎯 Generating negative examples for 'no object found' training...")
    all_filenames = set()
    for task_imgs in all_task_images.values():
        all_filenames.update(task_imgs)

    for task_id in range(len(TASK_NAMES)):
        if task_id in TASKS_NEEDING_CUSTOM_DATA:
            continue  # Skip tasks that need custom data

        relevant_cat_ids = set(TASK_TO_COCO_CATS[task_id])
        # Images containing task-relevant objects
        positive_filenames = set(all_task_images.get(task_id, []))

        # Negative candidates: images with NO task-relevant objects
        negative_candidates = [
            fname for fname in all_filenames
            if fname not in positive_filenames
        ]

        # Sample up to max_negatives_per_task
        if len(negative_candidates) > max_negatives_per_task:
            negative_candidates = random.sample(negative_candidates, max_negatives_per_task)

        for fname in negative_candidates:
            stem = Path(fname).stem
            lbl_path = Path(out_lbl_dir) / f"{stem}_neg_t{task_id}.txt"
            # Empty label file = no objects for this task
            with open(lbl_path, "w") as lf:
                pass  # Empty file — no bounding boxes

        print(f"  Task {task_id:2d} ({TASK_NAMES[task_id][:25]:<25s}): "
              f"{len(negative_candidates):4d} negative examples")


# ── Dataset YAML ───────────────────────────────────────────────────────────────

def create_dataset_yaml(out_path, train_img_dir, val_img_dir):
    data = {
        "path":  str(Path(out_path).parent.parent.resolve()),
        "train": str(Path(train_img_dir).resolve()),
        "val":   str(Path(val_img_dir).resolve()),
        "nc":    len(TASK_NAMES),
        "names": {i: name for i, name in enumerate(TASK_NAMES)},
    }
    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"\n✅ Dataset YAML saved → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = PATHS

    # ── Train set ──────────────────────────────────────────────────────────────
    train_task_images = filter_and_convert(
        ann_json_path = p["train_ann"],
        img_src_dir   = p["train_images"],
        out_img_dir   = p["filtered_train"] + "/images",
        out_lbl_dir   = p["filtered_train"] + "/labels",
        split         = "train",
    )

    # ── Val set ────────────────────────────────────────────────────────────────
    val_task_images = filter_and_convert(
        ann_json_path = p["val_ann"],
        img_src_dir   = p["val_images"],
        out_img_dir   = p["filtered_val"] + "/images",
        out_lbl_dir   = p["filtered_val"] + "/labels",
        split         = "val",
    )

    # ── Negative examples ──────────────────────────────────────────────────────
    generate_negative_examples(
        train_task_images,
        out_lbl_dir = p["filtered_train"] + "/labels",
    )

    # ── Dataset YAML ───────────────────────────────────────────────────────────
    create_dataset_yaml(
        out_path      = p["dataset_yaml"],
        train_img_dir = p["filtered_train"] + "/images",
        val_img_dir   = p["filtered_val"]   + "/images",
    )

    print("\n🎯 Dataset preparation complete!")
    print(f"   Tasks: {len(TASK_NAMES)}")
    print(f"   Train: {p['filtered_train']}")
    print(f"   Val:   {p['filtered_val']}")
    print(f"   YAML:  {p['dataset_yaml']}")
    print("\n⚠️  Reminder: Tasks [3,6,12,13] (fire/dig/extinguish/carpet) need")
    print("   custom images — add them to filtered/train/images + labels manually.\n")


if __name__ == "__main__":
    main()
