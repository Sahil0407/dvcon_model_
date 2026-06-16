"""
Task Configuration for DVCon Task-Aware YOLOv8n
Maps 14 tasks to relevant COCO class IDs and defines task metadata.
"""

# ── 14 DVCon Tasks ─────────────────────────────────────────────────────────────
TASK_NAMES = [
    "step on something",        # 0
    "sit comfortably",          # 1
    "place flowers",            # 2
    "get potatoes out of fire", # 3
    "water plant",              # 4
    "get lemon out of tea",     # 5
    "dig hole",                 # 6
    "open bottle of beer",      # 7
    "open parcel",              # 8
    "serve wine",               # 9
    "pour sugar",               # 10
    "smear butter",             # 11
    "extinguish fire",          # 12
    "pound carpet",             # 13
]

# ── COCO Category IDs (1-indexed as in COCO JSON) ──────────────────────────────
# Full list: https://cocodataset.org/#explore
COCO_CATEGORIES = {
    "person":       1,
    "bottle":       44,
    "wine glass":   46,
    "cup":          47,
    "fork":         48,
    "knife":        49,
    "spoon":        50,
    "bowl":         51,
    "chair":        62,
    "couch":        63,
    "potted plant": 64,
    "vase":         75,
    "scissors":     76,
    "book":         84,
    "laptop":       73,
    "remote":       65,
    "cell phone":   77,
    "umbrella":     25,
    "handbag":      31,
    "suitcase":     33,
    "sports ball":  37,
    "baseball bat": 39,
    "baseball glove": 40,
    "skateboard":   41,
    "surfboard":    42,
    "tennis racket":43,
}

# ── Task → Relevant COCO Category IDs ──────────────────────────────────────────
# These determine which COCO images get pulled for each task
TASK_TO_COCO_CATS = {
    0:  [1],                        # step on something  → person
    1:  [1, 62, 63],                # sit comfortably    → person, chair, couch
    2:  [1, 64, 75],                # place flowers      → person, potted plant, vase
    3:  [1],                        # get potatoes/fire  → person (custom images needed)
    4:  [1, 64],                    # water plant        → person, potted plant
    5:  [1, 47, 46],                # get lemon/tea      → person, cup, wine glass
    6:  [1],                        # dig hole           → person (custom needed)
    7:  [1, 44],                    # open bottle beer   → person, bottle
    8:  [1, 33],                    # open parcel        → person, suitcase/box
    9:  [1, 46, 44],                # serve wine         → person, wine glass, bottle
    10: [1, 51, 50],                # pour sugar         → person, bowl, spoon
    11: [1, 49],                    # smear butter       → person, knife
    12: [1],                        # extinguish fire    → person (custom needed)
    13: [1],                        # pound carpet       → person (custom needed)
}

# ── Tasks needing custom/supplemental images (not well-covered by COCO) ────────
TASKS_NEEDING_CUSTOM_DATA = [3, 6, 12, 13]
TASKS_WELL_COVERED_BY_COCO = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11]

# ── Model & Training Config ────────────────────────────────────────────────────
MODEL_CONFIG = {
    "base_model":       "yolov8n.pt",       # YOLOv8 Nano pretrained
    "num_tasks":        14,
    "num_classes":      14,                 # one class per task for task-aware detection
    "img_size":         640,
    "batch_size":       16,
    "epochs":           100,
    "lr0":              0.01,
    "lrf":              0.01,
    "warmup_epochs":    3,
    "device":           "cuda",             # or "cpu"
    "workers":          4,
    "patience":         20,                 # early stopping
    "save_period":      10,
    "project":          "dvcon_task_aware",
    "name":             "yolov8n_film",
}

# ── FiLM Injection Config ──────────────────────────────────────────────────────
# YOLOv8n (nano) neck channel sizes after width_multiple=0.25 scaling
# P3/P4/P5 are the correct injection points — they are the neck outputs
# that feed directly into the detection head. Conditioning here allows
# the task embedding to modulate features at ALL scale levels simultaneously.
#
# Why NOT backbone layers:
#   - Backbone features are generic (task-agnostic) — better to leave them intact
#   - Injecting early wastes compute on features that get re-aggregated in neck
#   - Research (YOLO-World, GLIP) confirms neck is the optimal fusion point
#
# Why NOT detection head only:
#   - Head-level conditioning lacks spatial context
#   - Cannot guide WHICH features to emphasize — only adjusts final scores
FILM_CONFIG = {
    "text_encoder":     "all-MiniLM-L6-v2",  # 384-dim, lightweight
    "embed_dim":        384,
    "injection_layers": {
        # layer_name_in_yolov8n : feature_channels
        "model.model.15":   64,              # P3 small-scale features (stride 8)
        "model.model.18":  128,              # P4 medium-scale features (stride 16)
        "model.model.21":  256,              # P5 large-scale features (stride 32)
    },
    # Hidden dims scaled proportionally to feature channels:
    # P3 (64ch)  → hidden=64   (1x feature dim)
    # P4 (128ch) → hidden=128  (1x feature dim)
    # P5 (256ch) → hidden=128  (0.5x — prevents overfitting on small dataset)
    "film_hidden_dims": {
        15: 64,    # P3 — small objects need precise, lightweight modulation
        18: 128,   # P4 — balanced capacity for medium objects
        21: 128,   # P5 — cap at 128 to avoid overfitting (256→128→256 would be too large)
    },
    "dropout":          0.1,
}

# ── Paths ──────────────────────────────────────────────────────────────────────
PATHS = {
    "coco_root":        "./data/coco",
    "train_images":     "./data/coco/images/train2017",
    "val_images":       "./data/coco/images/val2017",
    "train_ann":        "./data/coco/annotations/instances_train2017.json",
    "val_ann":          "./data/coco/annotations/instances_val2017.json",
    "filtered_train":   "./data/filtered/train",
    "filtered_val":     "./data/filtered/val",
    "task_embeddings":  "./data/task_embeddings.pt",
    "dataset_yaml":     "./configs/dataset.yaml",
    "output":           "./runs/",
}
