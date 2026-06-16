"""
Training Script — DVCon Task-Aware YOLOv8n with FiLM Injection (Batch-by-Task)
──────────────────────────────────────────────────────────────────────────────
3-Phase Training:
    Phase 1  │ Freeze backbone + neck, train FiLM layers only     (epochs 1-10)
    Phase 2  │ Unfreeze neck, train neck + FiLM jointly           (epochs 11-50)
    Phase 3  │ Full fine-tune (all layers + FiLM, low LR)         (epochs 51-100)

Key Design: Batch-by-Task
    Each batch contains images from a SINGLE task. The FiLM conditioner
    receives one task embedding for the entire batch, ensuring consistent
    conditioning. When no task-relevant objects exist, the model learns
    to output "no object found" (empty detection set).

v8DetectionLoss Input Format:
    Labels must be [batch_idx, class_id, cx, cy, w, h] (6 columns).
    Our dataset returns [task_id, cx, cy, w, h] (5 columns), so we
    prepend the batch index during collation.

Run:
    python train.py [--resume path/to/checkpoint.pt]
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

from configs.task_config import MODEL_CONFIG, FILM_CONFIG, PATHS, TASK_NAMES
from models.film_injection import FiLMHookManager
from utils.task_dataset import build_dataloaders
from utils.task_embeddings import generate_task_embeddings, load_task_embeddings


# ── Training Phases ────────────────────────────────────────────────────────────

PHASES = [
    {"name": "Phase1_FiLM_only",     "epochs": (1,  10),  "lr": 1e-3,  "freeze_backbone": True,  "freeze_neck": True},
    {"name": "Phase2_Neck_and_FiLM", "epochs": (11, 50),  "lr": 5e-4,  "freeze_backbone": True,  "freeze_neck": False},
    {"name": "Phase3_Full_finetune", "epochs": (51, 100), "lr": 1e-4,  "freeze_backbone": False, "freeze_neck": False},
]


def freeze_backbone(model: YOLO):
    """Freeze YOLOv8n backbone layers (0-9)."""
    for i, layer in enumerate(model.model.model):
        if i < 10:
            for p in layer.parameters():
                p.requires_grad_(False)
    print("🔒 Backbone frozen (layers 0-9)")


def freeze_neck(model: YOLO):
    """Freeze YOLOv8n neck layers (10-21)."""
    for i, layer in enumerate(model.model.model):
        if 10 <= i <= 21:
            for p in layer.parameters():
                p.requires_grad_(False)
    print("🔒 Neck frozen (layers 10-21)")


def unfreeze_all(model: YOLO):
    for p in model.model.parameters():
        p.requires_grad_(True)
    print("🔓 All YOLO layers unfrozen")


def get_phase(epoch: int) -> dict:
    for phase in PHASES:
        lo, hi = phase["epochs"]
        if lo <= epoch <= hi:
            return phase
    return PHASES[-1]


def format_labels_for_loss(labels_list, device):
    """
    Convert dataset labels to v8DetectionLoss format.

    v8DetectionLoss expects targets as [batch_idx, class_id, cx, cy, w, h].
    Our labels are [task_id, cx, cy, w, h] per image.

    Since all images in a batch share the same task_id, we prepend
    the batch index to each label row.

    Parameters
    ----------
    labels_list : list of [N_i, 5] tensors
        Raw labels from dataset (task_id, cx, cy, w, h).
    device : torch.device

    Returns
    -------
    targets : [M, 6] tensor  (batch_idx, class_id, cx, cy, w, h)
    """
    targets = []
    for batch_idx, labels in enumerate(labels_list):
        if labels.numel() == 0:
            continue
        labels = labels.to(device)
        # labels: [N, 5] — (task_id, cx, cy, w, h)
        # Prepend batch_idx column
        batch_col = torch.full((labels.shape[0], 1), batch_idx, device=device)
        targets.append(torch.cat([batch_col, labels], dim=1))

    if targets:
        return torch.cat(targets, dim=0)
    else:
        return torch.zeros(0, 6, device=device)


# ── Checkpoint ─────────────────────────────────────────────────────────────────

def save_checkpoint(epoch, yolo_model, film_manager, optimizer, loss, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f"checkpoint_epoch{epoch:04d}.pt"
    torch.save({
        "epoch":        epoch,
        "yolo_state":   yolo_model.model.state_dict(),
        "film_state":   film_manager.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "loss":         loss,
    }, path)
    return path


def load_checkpoint(path, yolo_model, film_manager, optimizer):
    ck = torch.load(path, map_location="cpu")
    yolo_model.model.load_state_dict(ck["yolo_state"])
    film_manager.load_state_dict(ck["film_state"])
    optimizer.load_state_dict(ck["optimizer"])
    print(f"📂 Resumed from epoch {ck['epoch']} (loss={ck['loss']:.4f})")
    return ck["epoch"]


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train(resume_path=None):
    device  = torch.device(MODEL_CONFIG["device"] if torch.cuda.is_available() else "cpu")
    run_dir = Path(PATHS["output"]) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🚀 DVCon Task-Aware YOLOv8n Training (Batch-by-Task)")
    print(f"   Device : {device}")
    print(f"   Run dir: {run_dir}\n")

    # ── 1. Task Embeddings ─────────────────────────────────────────────────────
    emb_path = PATHS["task_embeddings"]
    if not Path(emb_path).exists():
        print("Generating task embeddings...")
        generate_task_embeddings(save_path=emb_path)
    task_embs_all = load_task_embeddings(emb_path, device=str(device))  # [14, 384]

    # ── 2. Data Loaders ────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_img_dir   = PATHS["filtered_train"] + "/images",
        train_lbl_dir   = PATHS["filtered_train"] + "/labels",
        val_img_dir     = PATHS["filtered_val"]   + "/images",
        val_lbl_dir     = PATHS["filtered_val"]   + "/labels",
        embeddings_path = emb_path,
        batch_size      = MODEL_CONFIG["batch_size"],
        num_workers     = MODEL_CONFIG["workers"],
        img_size        = MODEL_CONFIG["img_size"],
    )

    # ── 3. Model Setup ─────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_CONFIG['base_model']} ...")
    yolo_model = YOLO(MODEL_CONFIG["base_model"])
    yolo_model.model = yolo_model.model.to(device)

    # FiLM hook manager
    film_manager = FiLMHookManager(yolo_model)
    film_manager.to(device)
    film_manager.register_hooks()

    # ── 4. Optimizer ───────────────────────────────────────────────────────────
    # Start with only FiLM params; expand per phase
    optimizer = optim.AdamW(
        list(film_manager.film_parameters()),
        lr=PHASES[0]["lr"],
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = MODEL_CONFIG["epochs"],
        eta_min = 1e-6,
    )

    # ── 5. Resume ──────────────────────────────────────────────────────────────
    start_epoch = 1
    if resume_path:
        start_epoch = load_checkpoint(resume_path, yolo_model, film_manager, optimizer) + 1

    # ── 6. Loss Function ───────────────────────────────────────────────────────
    from ultralytics.utils.loss import v8DetectionLoss
    yolo_loss_fn = v8DetectionLoss(yolo_model.model)

    best_val_loss   = float("inf")
    patience_counter = 0

    # ── 7. Epoch Loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, MODEL_CONFIG["epochs"] + 1):
        phase = get_phase(epoch)

        # Apply phase-specific freezing / unfreezing
        if epoch == PHASES[0]["epochs"][0]:
            freeze_backbone(yolo_model)
            freeze_neck(yolo_model)
            optimizer.param_groups[0]["lr"] = PHASES[0]["lr"]

        elif epoch == PHASES[1]["epochs"][0]:
            freeze_backbone(yolo_model)
            for i, layer in enumerate(yolo_model.model.model):
                if 10 <= i <= 21:
                    for p in layer.parameters():
                        p.requires_grad_(True)
            # Add neck params to optimizer
            optimizer.add_param_group({
                "params": [p for i, l in enumerate(yolo_model.model.model)
                           if 10 <= i <= 21 for p in l.parameters()],
                "lr": PHASES[1]["lr"],
            })
            print("🔓 Neck unfrozen — neck + FiLM training begins")

        elif epoch == PHASES[2]["epochs"][0]:
            unfreeze_all(yolo_model)
            optimizer.add_param_group({
                "params": [p for i, l in enumerate(yolo_model.model.model)
                           if i < 10 for p in l.parameters()],
                "lr": PHASES[2]["lr"],
            })
            print("🔓 Full fine-tune phase begins")

        # ── Train ──────────────────────────────────────────────────────────────
        yolo_model.model.train()
        film_manager.train()
        total_train_loss = 0.0

        for batch_idx, (images, labels_list, task_id) in enumerate(train_loader):
            images = images.to(device)

            # Batch-by-task: all images share the same task embedding
            task_emb = task_embs_all[task_id].unsqueeze(0).expand(images.shape[0], -1)
            task_emb = task_emb.to(device)

            # Inject task embedding for this batch before forward pass
            film_manager.set_task_embedding(task_emb)

            optimizer.zero_grad()

            # Forward pass through YOLOv8n (hooks apply FiLM automatically)
            preds = yolo_model.model(images)

            # Format labels: prepend batch_idx for v8DetectionLoss
            targets = format_labels_for_loss(labels_list, device)

            # Compute YOLO detection loss
            loss, loss_items = yolo_loss_fn(preds, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(yolo_model.model.parameters()) +
                list(film_manager.film_parameters()),
                max_norm=10.0
            )
            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch:3d} | Phase: {phase['name']} | "
                      f"Task: {task_id:2d} ({TASK_NAMES[task_id][:20]}) | "
                      f"Batch {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} "
                      f"(box={loss_items[0]:.3f} cls={loss_items[1]:.3f} dfl={loss_items[2]:.3f})")

        # ── Validate ───────────────────────────────────────────────────────────
        yolo_model.model.eval()
        film_manager.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for images, labels_list, task_id in val_loader:
                images = images.to(device)
                task_emb = task_embs_all[task_id].unsqueeze(0).expand(images.shape[0], -1)
                task_emb = task_emb.to(device)
                film_manager.set_task_embedding(task_emb)
                preds = yolo_model.model(images)
                targets = format_labels_for_loss(labels_list, device)
                loss, _ = yolo_loss_fn(preds, targets)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val   = total_val_loss   / len(val_loader)
        scheduler.step()

        print(f"\n📊 Epoch {epoch:3d} Summary | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}\n")

        # ── Save ───────────────────────────────────────────────────────────────
        if epoch % MODEL_CONFIG["save_period"] == 0:
            ck = save_checkpoint(epoch, yolo_model, film_manager, optimizer,
                                 avg_val, run_dir / "checkpoints")
            print(f"💾 Checkpoint saved → {ck}")

        # ── Best model ─────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            best_path = run_dir / "best.pt"
            torch.save({
                "epoch":      epoch,
                "yolo_state": yolo_model.model.state_dict(),
                "film_state": film_manager.state_dict(),
                "val_loss":   best_val_loss,
            }, best_path)
            print(f"🏆 New best model (val_loss={best_val_loss:.4f}) → {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= MODEL_CONFIG["patience"]:
                print(f"⏹  Early stopping at epoch {epoch} (no improvement for {MODEL_CONFIG['patience']} epochs)")
                break

    film_manager.remove_hooks()
    print(f"\n✅ Training complete! Best model: {run_dir}/best.pt")
    return str(run_dir / "best.pt")


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DVCon Task-Aware YOLOv8n Trainer")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume from")
    args = parser.parse_args()
    train(resume_path=args.resume)
