"""
Inference Script — DVCon Task-Aware YOLOv8n (Batch-by-Task)
─────────────────────────────────────────────────────────────
Three modes:
    1. Task-specific (by name)  : specify a task string → model detects objects for THAT task
    2. Task-specific (by ID)    : specify a task ID 0-13 → model detects objects for THAT task
    3. All-tasks                : run all 14 task embeddings, merge results

The model outputs "no object found" when no detections pass the confidence
threshold — this is the expected behavior when the task-relevant objects
are not present in the image.

Supports arbitrary task text at runtime via Sentence-BERT fallback:
    python inference.py --image test.jpg --task "pick up the red cup"
    (will encode "pick up the red cup" on-the-fly if not in the 14 predefined tasks)

Run:
    python inference.py --image path/to/image.jpg --checkpoint best.pt --task "open bottle of beer"
    python inference.py --image path/to/image.jpg --checkpoint best.pt --task_id 7
    python inference.py --image path/to/image.jpg --checkpoint best.pt --all_tasks
    python inference.py --image path/to/image.jpg --checkpoint best.pt --task "pick up the red cup"  # arbitrary text
"""

import argparse
import torch
import cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes
from ultralytics.utils.nms import non_max_suppression

from typing import Optional, List
from configs.task_config import TASK_NAMES, PATHS, MODEL_CONFIG, FILM_CONFIG
from models.film_injection import FiLMHookManager
from utils.task_embeddings import load_task_embeddings


# ── Colors per task ────────────────────────────────────────────────────────────
PALETTE = [
    (255, 56,  56),  (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72,  249, 10),  (146, 204, 23),  (61,  219, 134),
    (26,  147, 52),  (0,   212, 187), (44,  153, 168),  (0,   194, 255),
    (52,  69,  147), (100, 115, 255),
]
NO_OBJECT_COLOR = (128, 128, 128)  # Gray for "no object found"


def draw_boxes(image, detections, task_id, task_name, conf_thresh=0.25):
    """Draw detected bounding boxes on the image."""
    color = PALETTE[task_id % len(PALETTE)] if task_id < len(TASK_NAMES) else NO_OBJECT_COLOR
    for det in detections:
        x1, y1, x2, y2, conf, cls = det[:6]
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"[T{task_id}] {task_name[:20]} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 4), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image


def draw_no_object_banner(image, task_id, task_name):
    """Draw a banner indicating no object was found for this task."""
    h, w = image.shape[:2]
    banner_text = f"Task {task_id}: {task_name} — NO OBJECT FOUND"
    color = NO_OBJECT_COLOR

    # Draw semi-transparent banner at top
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    cv2.putText(image, banner_text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def load_model(checkpoint_path: str, device: torch.device):
    """Load YOLOv8n + FiLM weights from checkpoint."""
    yolo_model = YOLO(MODEL_CONFIG["base_model"])
    yolo_model.model = yolo_model.model.to(device)

    film_manager = FiLMHookManager(yolo_model)
    film_manager.to(device)
    film_manager.register_hooks()

    ck = torch.load(checkpoint_path, map_location=device)
    yolo_model.model.load_state_dict(ck["yolo_state"])
    film_manager.load_state_dict(ck["film_state"])

    yolo_model.model.eval()
    film_manager.eval()
    print(f"✅ Loaded checkpoint (epoch {ck.get('epoch','?')}) from {checkpoint_path}")
    return yolo_model, film_manager


_sentence_encoder_cache = None


def encode_arbitrary_task(task_text: str, device: torch.device) -> torch.Tensor:
    """
    Encode an arbitrary task description using Sentence-BERT on-the-fly.
    Returns a [1, 384] embedding tensor.

    This allows the model to generalize to unseen task descriptions
    beyond the 14 predefined tasks.
    """
    global _sentence_encoder_cache
    if _sentence_encoder_cache is None:
        from sentence_transformers import SentenceTransformer
        _sentence_encoder_cache = SentenceTransformer(FILM_CONFIG["text_encoder"])
    encoder = _sentence_encoder_cache
    embedding = encoder.encode(
        task_text,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    return embedding.unsqueeze(0).to(device)  # [1, 384]


def resolve_task_embedding(
    task_text: Optional[str],
    task_id: Optional[int],
    task_embs: torch.Tensor,
    device: torch.device,
):
    """
    Resolve task specification to (embedding, name, task_id).

    Priority:
      1. If task_text matches a predefined task name → use precomputed embedding
      2. If task_id is provided → use precomputed embedding
      3. Otherwise → encode arbitrary text via Sentence-BERT fallback
    """
    if task_id is not None:
        assert 0 <= task_id < len(TASK_NAMES), f"Invalid task_id: {task_id}"
        return task_embs[task_id].unsqueeze(0), TASK_NAMES[task_id], task_id

    if task_text:
        # Check if it matches a predefined task
        name_lower = task_text.lower().strip()
        for i, name in enumerate(TASK_NAMES):
            if name.lower() == name_lower:
                return task_embs[i].unsqueeze(0), name, i

        # Arbitrary text — encode on-the-fly
        print(f"  📝 Encoding arbitrary task: \"{task_text}\"")
        emb = encode_arbitrary_task(task_text, device)
        return emb, task_text, -1  # task_id=-1 for arbitrary tasks

    raise ValueError("Provide --task, --task_id, or --all_tasks")


@torch.no_grad()
def run_inference(
    image_path:    str,
    checkpoint:    str,
    task_ids:      Optional[List[int]] = None,
    task_text:     Optional[str] = None,
    conf_thresh:   float = 0.25,
    iou_thresh:    float = 0.45,
    img_size:      int   = 640,
    save_dir:      str   = "./inference_out",
):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task_embs = load_task_embeddings(PATHS["task_embeddings"], device=str(device))  # [14, 384]

    yolo_model, film_manager = load_model(checkpoint, device)

    # ── Load & preprocess image ────────────────────────────────────────────────
    orig_img = cv2.imread(image_path)
    assert orig_img is not None, f"Cannot read image: {image_path}"
    h0, w0 = orig_img.shape[:2]

    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (img_size, img_size))
    img_t   = torch.from_numpy(img_res).permute(2, 0, 1).float().div(255.0)
    img_t   = img_t.unsqueeze(0).to(device)  # [1, 3, H, W]

    output_img = orig_img.copy()
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_dets = []
    results_summary = []

    # ── Determine tasks to run ─────────────────────────────────────────────────
    if task_ids is not None:
        tasks_to_run = [(task_embs[i].unsqueeze(0), TASK_NAMES[i], i) for i in task_ids]
    elif task_text:
        emb, name, tid = resolve_task_embedding(task_text, None, task_embs, device)
        tasks_to_run = [(emb, name, tid)]
    else:
        raise ValueError("Provide task_ids or task_text")

    for emb, task_name, task_id in tasks_to_run:
        # Inject task embedding for this forward pass
        film_manager.set_task_embedding(emb)

        # Forward
        preds = yolo_model.model(img_t)  # raw YOLO output

        # NMS
        dets = non_max_suppression(
            preds, conf_thresh, iou_thresh,
            classes=None, agnostic=False, max_det=50
        )[0]  # [N, 6]  xyxy conf cls

        if dets is not None and len(dets):
            # Scale boxes back to original image size
            dets[:, :4] = scale_boxes(img_t.shape[2:], dets[:, :4], orig_img.shape).round()
            output_img = draw_boxes(output_img, dets.cpu().numpy(), task_id, task_name, conf_thresh)
            all_dets.append((task_id, task_name, dets.cpu()))
            results_summary.append(f"  Task [{task_id:2d}] {task_name:<30s} → {len(dets)} detection(s)")
        else:
            # No object found — draw banner
            if task_id >= 0:
                output_img = draw_no_object_banner(output_img, task_id, task_name)
            results_summary.append(f"  Task [{task_id:2d}] {task_name:<30s} → ⚪ NO OBJECT FOUND")

    # Print results
    print("\n" + "=" * 60)
    print("  INFERENCE RESULTS")
    print("=" * 60)
    for line in results_summary:
        print(line)
    print("=" * 60)

    # Save result
    stem     = Path(image_path).stem
    out_path = Path(save_dir) / f"{stem}_task_aware.jpg"
    cv2.imwrite(str(out_path), output_img)
    print(f"\n💾 Result saved → {out_path}")

    film_manager.remove_hooks()
    return all_dets


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVCon Task-Aware YOLOv8n Inference")
    parser.add_argument("--image",      required=True,  help="Input image path")
    parser.add_argument("--checkpoint", required=True,  help="Path to best.pt checkpoint")
    parser.add_argument("--task",       default=None,   help='Task name e.g. "open bottle of beer" or arbitrary text')
    parser.add_argument("--task_id",    type=int,       default=None, help="Task ID 0-13")
    parser.add_argument("--all_tasks",  action="store_true",          help="Run all 14 tasks")
    parser.add_argument("--conf",       type=float,     default=0.25)
    parser.add_argument("--iou",        type=float,     default=0.45)
    parser.add_argument("--save_dir",   default="./inference_out")
    args = parser.parse_args()

    if args.all_tasks:
        task_ids = list(range(len(TASK_NAMES)))
        task_text = None
    elif args.task_id is not None:
        task_ids = [args.task_id]
        task_text = None
    elif args.task:
        task_ids = None
        task_text = args.task
    else:
        raise ValueError("Provide --task, --task_id, or --all_tasks")

    run_inference(
        image_path  = args.image,
        checkpoint  = args.checkpoint,
        task_ids    = task_ids,
        task_text   = task_text,
        conf_thresh = args.conf,
        iou_thresh  = args.iou,
        save_dir    = args.save_dir,
    )
