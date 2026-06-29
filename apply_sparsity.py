"""
Channel-Wise 2:4 Structured Sparsity — DVCon TaskAwareYOLO + FiLM
══════════════════════════════════════════════════════════════════

WHAT IS 2:4 SPARSITY?
──────────────────────
2:4 structured sparsity means: in every group of 4 weights, exactly 2 must be
zero (50% sparsity). NVIDIA Ampere+ sparse tensor cores execute this natively
at ~2× speedup with zero accuracy loss when applied post-training.

STANDARD 2:4 vs. THIS IMPLEMENTATION
──────────────────────────────────────
Standard 2:4 (NVIDIA default, cuSPARSELt):
    Conv weight shape : [out_c, in_c, kH, kW]
    Groups of 4 taken : along the *flattened* (in_c × kH × kW) dimension
                        i.e. consecutive spatial pixels within a single
                        input channel are grouped together.
    Visualised (one output filter, one input channel, spatial row):
        [w00 w01 w02 w03 | w04 w05 w06 w07 | ...]
         ──── group 0 ────  ──── group 1 ────

This implementation — Channel-Wise 2:4:
    Groups of 4 taken : across *input channels* at a FIXED spatial position.
    For each (out_c, kH_pos, kW_pos), stack in_c values and group as:
        [ch0 ch1 ch2 ch3 | ch4 ch5 ch6 ch7 | ...]
         ──── group 0 ────  ──── group 1 ────
    Sparsity encodes "which channels matter at this pixel" rather than
    "which spatial positions matter within this channel".

    Tensor mechanics:
        [out_c, in_c, kH, kW]
          → permute → [out_c, kH, kW, in_c]
          → reshape  → [out_c*kH*kW,  in_c//4, 4]
          → top-2 mask on dim=-1
          → reshape  → [out_c, kH, kW, in_c]
          → permute  → [out_c, in_c, kH, kW]

    For Linear [out_f, in_f]:
        → reshape → [out_f, in_f//4, 4]
        → top-2 mask on dim=-1
        → reshape → [out_f, in_f]

LAYERS PRUNED
─────────────
  • All nn.Conv2d in YOLOv8n backbone + neck (weight only, not bias)
  • FiLM gamma_net / beta_net Linear layers (weight only, not bias)
  • Layers where in_c (or in_f) is not divisible by 4 → SKIPPED with warning

USAGE
─────
    # Basic
    python apply_sparsity.py --checkpoint best.pt --output sparse_best.pt

    # Verify sparsity of every pruned layer after applying
    python apply_sparsity.py --checkpoint best.pt --output sparse_best.pt --verify

    # Dry-run: print which layers will be pruned without saving
    python apply_sparsity.py --checkpoint best.pt --dry_run

    # Apply only to FiLM layers (skip YOLO backbone/neck)
    python apply_sparsity.py --checkpoint best.pt --output sparse_best.pt --film_only

    # Apply only to YOLO layers (skip FiLM)
    python apply_sparsity.py --checkpoint best.pt --output sparse_best.pt --yolo_only

IMPORTANT NOTES
───────────────
  1. This is MAGNITUDE-BASED pruning (top-2 by |w| per group of 4).
     Apply AFTER full training for best accuracy.

  2. The script rewrites the weight tensors in-place in the state_dict.
     Original checkpoint is NOT modified; a new file is written.

  3. For NVIDIA sparse tensor core execution, load the sparse checkpoint
     and use torch.backends.cuda.matmul.allow_tf32 + cuSPARSELt or
     APEX's ASP library for actual sparse kernel dispatch.

  4. The mask is NOT stored separately — zeros are baked into weights.
     If you need to resume fine-tuning with a fixed mask, you must
     regenerate the mask each forward pass (see: mask-based training).
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO

from configs.task_config import MODEL_CONFIG, FILM_CONFIG
from models.film_injection import FiLMHookManager


# ══════════════════════════════════════════════════════════════════════════════
# Core mask computation
# ══════════════════════════════════════════════════════════════════════════════

def channel_wise_2_4_mask_conv(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute a binary 2:4 mask for a Conv2d weight tensor.

    Grouping axis: input channels (dim=1) at each fixed spatial position.

    Parameters
    ----------
    weight : torch.Tensor  shape [out_c, in_c, kH, kW]

    Returns
    -------
    mask : torch.Tensor  shape [out_c, in_c, kH, kW]  — 1=keep, 0=zero
           None if in_c % 4 != 0 (caller should skip this layer)
    """
    out_c, in_c, kH, kW = weight.shape

    if in_c % 4 != 0:
        return None  # Cannot form complete groups; caller skips

    # Step 1: permute so in_c is the last dim  →  [out_c, kH, kW, in_c]
    w_perm = weight.permute(0, 2, 3, 1).contiguous()

    # Step 2: reshape into groups of 4 along the channel axis
    #         → [out_c * kH * kW,  in_c // 4,  4]
    n_positions = out_c * kH * kW
    w_groups = w_perm.view(n_positions, in_c // 4, 4)

    # Step 3: find the 2 largest magnitudes in each group of 4
    abs_groups = w_groups.abs()
    _, top2_indices = torch.topk(abs_groups, k=2, dim=-1)  # [N, G, 2]

    # Step 4: scatter 1s into a zero mask at the top-2 positions
    mask_groups = torch.zeros_like(w_groups)
    mask_groups.scatter_(-1, top2_indices, 1.0)

    # Step 5: reshape back  →  [out_c, kH, kW, in_c]
    mask_perm = mask_groups.view(out_c, kH, kW, in_c)

    # Step 6: permute back to original layout  →  [out_c, in_c, kH, kW]
    mask = mask_perm.permute(0, 3, 1, 2).contiguous()

    return mask


def channel_wise_2_4_mask_linear(weight: torch.Tensor) -> torch.Tensor:
    """
    Compute a binary 2:4 mask for a Linear weight tensor.

    For Linear [out_f, in_f], groups of 4 are taken along in_f
    (input features = "channels" for the linear analogy).

    Parameters
    ----------
    weight : torch.Tensor  shape [out_f, in_f]

    Returns
    -------
    mask : torch.Tensor  shape [out_f, in_f]  — 1=keep, 0=zero
           None if in_f % 4 != 0
    """
    out_f, in_f = weight.shape

    if in_f % 4 != 0:
        return None

    # [out_f, in_f // 4, 4]
    w_groups = weight.view(out_f, in_f // 4, 4)
    abs_groups = w_groups.abs()
    _, top2_indices = torch.topk(abs_groups, k=2, dim=-1)

    mask_groups = torch.zeros_like(w_groups)
    mask_groups.scatter_(-1, top2_indices, 1.0)

    mask = mask_groups.view(out_f, in_f)
    return mask


def apply_mask(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero out weights where mask == 0. Returns a new tensor."""
    return weight * mask


# ══════════════════════════════════════════════════════════════════════════════
# Layer-level sparsity application
# ══════════════════════════════════════════════════════════════════════════════

def prune_state_dict_layer(
    state_dict: dict,
    key: str,
    mask_fn,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Apply channel-wise 2:4 sparsity to one weight tensor in a state dict.

    Parameters
    ----------
    state_dict : the model state dict (modified in-place)
    key        : the key of the weight tensor to prune
    mask_fn    : channel_wise_2_4_mask_conv or channel_wise_2_4_mask_linear
    dry_run    : if True, compute mask but do NOT write back
    verbose    : print per-layer results

    Returns
    -------
    info dict with keys: key, shape, skipped, sparsity_before, sparsity_after
    """
    w = state_dict[key]
    total_params = w.numel()
    sparsity_before = (w == 0).sum().item() / total_params * 100

    mask = mask_fn(w)

    if mask is None:
        if verbose:
            dim_name = "in_c" if w.dim() == 4 else "in_f"
            print(f"  ⚠️  SKIP  {key:<60s} shape={list(w.shape)}  "
                  f"({dim_name}={w.shape[1] if w.dim()==4 else w.shape[1]} not divisible by 4)")
        return {"key": key, "shape": list(w.shape), "skipped": True,
                "sparsity_before": sparsity_before, "sparsity_after": sparsity_before}

    w_sparse = apply_mask(w, mask)
    sparsity_after = (w_sparse == 0).sum().item() / total_params * 100

    if not dry_run:
        state_dict[key] = w_sparse

    if verbose:
        print(f"  ✅  {key:<60s} shape={list(w.shape)}  "
              f"sparsity: {sparsity_before:5.1f}% → {sparsity_after:5.1f}%")

    return {"key": key, "shape": list(w.shape), "skipped": False,
            "sparsity_before": sparsity_before, "sparsity_after": sparsity_after}


# ══════════════════════════════════════════════════════════════════════════════
# Full model sparsification
# ══════════════════════════════════════════════════════════════════════════════

def apply_channel_wise_sparsity(
    yolo_state: dict,
    film_state: dict,
    dry_run: bool = False,
    film_only: bool = False,
    yolo_only: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Apply channel-wise 2:4 sparsity to the YOLO + FiLM state dicts.

    Parameters
    ----------
    yolo_state : state_dict from yolo_model.model
    film_state : state_dict from film_manager.film_layers
    dry_run    : analyse only, do not modify weights
    film_only  : prune FiLM linear layers only
    yolo_only  : prune YOLO conv layers only
    verbose    : per-layer print

    Returns
    -------
    summary dict with stats per section
    """
    results = {"yolo": [], "film": []}

    # ── YOLO Conv layers ───────────────────────────────────────────────────────
    if not film_only:
        if verbose:
            print("\n" + "═" * 72)
            print("  YOLO Conv2d layers  (channel-wise 2:4 along input channels)")
            print("═" * 72)

        for key in sorted(yolo_state.keys()):
            # Only weight tensors of Conv2d (4-D, skip bias)
            if not key.endswith(".weight"):
                continue
            w = yolo_state[key]
            if w.dim() != 4:
                continue  # not a conv weight

            info = prune_state_dict_layer(
                yolo_state, key,
                mask_fn  = channel_wise_2_4_mask_conv,
                dry_run  = dry_run,
                verbose  = verbose,
            )
            results["yolo"].append(info)

    # ── FiLM Linear layers ─────────────────────────────────────────────────────
    if not yolo_only:
        if verbose:
            print("\n" + "═" * 72)
            print("  FiLM gamma_net / beta_net Linear layers")
            print("═" * 72)

        for key in sorted(film_state.keys()):
            if not key.endswith(".weight"):
                continue
            w = film_state[key]
            if w.dim() != 2:
                continue  # not a linear weight

            info = prune_state_dict_layer(
                film_state, key,
                mask_fn  = channel_wise_2_4_mask_linear,
                dry_run  = dry_run,
                verbose  = verbose,
            )
            results["film"].append(info)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Verification
# ══════════════════════════════════════════════════════════════════════════════

def verify_sparsity(state_dict: dict, label: str = "", is_conv: bool = True):
    """
    Verify that every prunable layer satisfies the 2:4 constraint:
    in every group of 4 along the channel axis, exactly ≤ 2 are non-zero.

    Prints a per-layer pass/fail and a global summary.
    """
    print(f"\n{'═'*72}")
    print(f"  VERIFICATION — {label}")
    print(f"{'═'*72}")
    all_pass = True

    for key in sorted(state_dict.keys()):
        if not key.endswith(".weight"):
            continue
        w = state_dict[key]

        if is_conv and w.dim() != 4:
            continue
        if not is_conv and w.dim() != 2:
            continue

        # Reconstruct the grouping used during pruning
        if is_conv:
            out_c, in_c, kH, kW = w.shape
            if in_c % 4 != 0:
                continue
            # Permute + reshape: [out_c*kH*kW, in_c//4, 4]
            w_perm   = w.permute(0, 2, 3, 1).contiguous()
            w_groups = w_perm.view(out_c * kH * kW, in_c // 4, 4)
        else:
            out_f, in_f = w.shape
            if in_f % 4 != 0:
                continue
            w_groups = w.view(out_f, in_f // 4, 4)

        # Count non-zeros per group (should be ≤ 2 everywhere)
        nnz_per_group = (w_groups != 0).sum(dim=-1)   # [N, G]
        max_nnz       = nnz_per_group.max().item()
        violations    = (nnz_per_group > 2).sum().item()
        total_groups  = nnz_per_group.numel()
        pass_flag     = violations == 0

        status = "✅ PASS" if pass_flag else "❌ FAIL"
        if not pass_flag:
            all_pass = False

        print(f"  {status}  {key:<55s}  "
              f"groups={total_groups:6d}  max_nnz/group={max_nnz}  violations={violations}")

    print(f"\n  {'All layers satisfy 2:4 ✅' if all_pass else 'Some layers violate 2:4 ❌'}")
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Apply channel-wise 2:4 structured sparsity to TaskAwareYOLO + FiLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to trained checkpoint (best.pt or checkpoint_epochXXXX.pt)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for sparse checkpoint (default: <checkpoint>_sparse.pt)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After pruning, verify the 2:4 constraint holds for all layers",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Analyse layers and report what would be pruned, but do NOT save",
    )
    parser.add_argument(
        "--film_only", action="store_true",
        help="Prune only FiLM linear layers (skip YOLO conv layers)",
    )
    parser.add_argument(
        "--yolo_only", action="store_true",
        help="Prune only YOLO conv layers (skip FiLM linear layers)",
    )
    args = parser.parse_args()

    if args.film_only and args.yolo_only:
        raise ValueError("--film_only and --yolo_only are mutually exclusive")

    ck_path = Path(args.checkpoint)
    if not ck_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")

    out_path = Path(args.output) if args.output else \
               ck_path.parent / (ck_path.stem + "_sparse.pt")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"\n📂 Loading checkpoint: {ck_path}")
    ck = torch.load(ck_path, map_location="cpu")

    yolo_state = ck["yolo_state"]
    film_state = ck["film_state"]

    print(f"   Epoch          : {ck.get('epoch', '?')}")
    print(f"   Val loss       : {ck.get('val_loss', ck.get('loss', '?'))}")
    print(f"   YOLO keys      : {len(yolo_state)}")
    print(f"   FiLM keys      : {len(film_state)}")

    if args.dry_run:
        print("\n⚙️  DRY RUN — weights will NOT be modified\n")

    # ── Apply sparsity ─────────────────────────────────────────────────────────
    results = apply_channel_wise_sparsity(
        yolo_state = yolo_state,
        film_state = film_state,
        dry_run    = args.dry_run,
        film_only  = args.film_only,
        yolo_only  = args.yolo_only,
        verbose    = True,
    )

    # ── Summary stats ──────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  SUMMARY")
    print("═" * 72)

    for section, label in [("yolo", "YOLO Conv"), ("film", "FiLM Linear")]:
        infos = results[section]
        if not infos:
            continue
        pruned  = [i for i in infos if not i["skipped"]]
        skipped = [i for i in infos if i["skipped"]]
        if pruned:
            avg_before = sum(i["sparsity_before"] for i in pruned) / len(pruned)
            avg_after  = sum(i["sparsity_after"]  for i in pruned) / len(pruned)
            total_params_pruned = sum(
                i["shape"][0] * i["shape"][1] * (i["shape"][2] if len(i["shape"]) > 2 else 1)
                               * (i["shape"][3] if len(i["shape"]) > 3 else 1)
                for i in pruned
            )
            print(f"\n  {label}")
            print(f"    Layers pruned   : {len(pruned)}")
            print(f"    Layers skipped  : {len(skipped)} (in_dim not divisible by 4)")
            print(f"    Avg sparsity    : {avg_before:.1f}% → {avg_after:.1f}%")
            print(f"    Total params    : {total_params_pruned:,}")

    # ── Verify ─────────────────────────────────────────────────────────────────
    if args.verify and not args.dry_run:
        if not args.film_only:
            verify_sparsity(yolo_state, label="YOLO Conv2d", is_conv=True)
        if not args.yolo_only:
            verify_sparsity(film_state, label="FiLM Linear", is_conv=False)

    # ── Save ───────────────────────────────────────────────────────────────────
    if not args.dry_run:
        sparse_ck = {
            **ck,                         # preserve epoch, val_loss, optimizer, etc.
            "yolo_state": yolo_state,
            "film_state": film_state,
            "sparsity": "channel_wise_2_4",
        }
        torch.save(sparse_ck, out_path)
        print(f"\n💾 Sparse checkpoint saved → {out_path}")
    else:
        print("\n⚙️  Dry run complete. No file written.")


if __name__ == "__main__":
    main()
