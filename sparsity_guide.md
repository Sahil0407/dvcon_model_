What standard 2:4 does vs. what you want:

Standard: for a conv weight [out_c, in_c, kH, kW], flatten the last three dims into one long vector and take groups of 4 along that flattened row. So groups span (channel, spatial) mixed together within a single output filter.
Your variant (channel-wise 2:4): at every fixed spatial position (kH, kW) for every output channel, take 4 consecutive input channels as a group. Group 0 = channels 0–3 at that pixel, group 1 = channels 4–7, etc. Sparsity is enforced across the channel stack, not along a spatial row.

Tensor mechanics: permute [out_c, in_c, kH, kW] → [out_c, kH, kW, in_c], reshape to [out_c*kH*kW, in_c//4, 4], apply top-2 mask on last dim, permute back. For Linear [out_f, in_f] it's the same idea: [out_f, in_f//4, 4].Created a file, read a fileCreated a file, read a fileHow channel-wise 2:4 differs from standard — concretely:
Suppose you have a 3×3 conv with in_c=64. At the spatial position (row=1, col=1) of one output filter, you have 64 weights — one per input channel. Standard 2:4 would group those 64 values alongside the adjacent spatial positions across the flattened row. Your variant groups those 64 values purely along the channel axis: channels 0–3 form group 0, channels 4–7 form group 1, etc. Within each group of 4, only the 2 largest-magnitude weights survive.

Key design decisions:

Layers skipped automatically: any conv or linear where in_c / in_f is not divisible by 4. This includes the YOLO detection head's final 1×1 convs which often have odd channel counts. A warning is printed for each.
Bias is never pruned: biases don't participate in NVIDIA's sparse format and pruning them offers no speedup.
Mask baked in, not stored separately: the zeros are written directly into the weight tensors. If you later want to fine-tune with a frozen mask (STE-style sparse training), you'd need to recompute the mask each step and re-apply — the script as-is is post-training only.
"sparsity": "channel_wise_2_4" is tagged into the checkpoint so you can distinguish it from a dense checkpoint when loading.


# 1. Dry run — see which layers get pruned and what sparsity they'd reach
python apply_sparsity.py --checkpoint runs/run_*/best.pt --dry_run

# 2. Apply and save
python apply_sparsity.py --checkpoint runs/run_*/best.pt --output sparse_best.pt

# 3. Apply + verify the 2:4 constraint holds mathematically
python apply_sparsity.py --checkpoint runs/run_*/best.pt --output sparse_best.pt --verify

# 4. Only prune FiLM layers (useful if you want to keep YOLO dense)
python apply_sparsity.py --checkpoint runs/run_*/best.pt --output sparse_best.pt --film_only
