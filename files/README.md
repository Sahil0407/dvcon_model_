# DVCon Task-Aware YOLOv8n with FiLM Injection (Batch-by-Task)

Object detection conditioned on **14 task embeddings** injected into YOLOv8 Nano's neck
via Feature-wise Linear Modulation (FiLM).

---

## Architecture

```
Task Description (text)
        │
   [Sentence-BERT]          ← all-MiniLM-L6-v2 (384-dim, ~22M params)
        │
  Task Embedding [1, 384]
        │
   ┌────▼──────────────────────────────────────┐
   │         FiLM Injection (3 points)         │
   │  γ(e) × feature_map + β(e)               │
   │  Layer 15 → P3 (64ch)   hidden=64        │
   │  Layer 18 → P4 (128ch)  hidden=128       │
   │  Layer 21 → P5 (256ch)  hidden=128       │
   └───────────────────────────────────────────┘
        │
  [YOLOv8n Detection Head]
        │
  Bounding Boxes + Task Class
```

### Why P3/P4/P5?

P3 (stride 8), P4 (stride 16), and P5 (stride 32) are the **neck outputs** that feed
directly into the detection head. Research (YOLO-World, GLIP) confirms this is the
optimal fusion point for task conditioning:
- Neck features are already multi-scale — conditioning here modulates ALL scales
- Backbone features are generic (task-agnostic) — better left intact
- Head-only conditioning lacks spatial context to guide detection

---

## Batch-by-Task Design

Each image is assigned a **single dominant task** (the task whose relevant objects
are most frequent in the image). During training, images are **grouped by task**
so each batch contains images from the same task.

**Benefits:**
1. FiLM conditioning is consistent within each batch
2. Model learns: "given task X, detect objects relevant to X"
3. When no relevant objects exist → model learns to output **"no object found"**
4. Negative examples (empty label files) explicitly train the no-object-found behavior

---

## Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Download COCO 2017
```bash
mkdir -p data/coco/images data/coco/annotations

# Images
wget -P data/coco/images http://images.cocodataset.org/zips/train2017.zip
wget -P data/coco/images http://images.cocodataset.org/zips/val2017.zip

# Annotations
wget -P data/coco/annotations \
  http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip data/coco/images/train2017.zip   -d data/coco/images/
unzip data/coco/images/val2017.zip     -d data/coco/images/
unzip data/coco/annotations/annotations_trainval2017.zip -d data/coco/
```

### 3. Prepare Dataset
```bash
python prepare_dataset.py
```
Filters COCO by task-relevant categories → assigns single dominant task per image.
Generates negative examples for "no object found" training.

⚠️ Tasks 3, 6, 12, 13 need custom images (fire/dig/extinguish/carpet).

### 4. Generate Task Embeddings (once)
```bash
python utils/task_embeddings.py
```
Saves `data/task_embeddings.pt` — 14 × 384 tensor.

### 5. Train
```bash
python train.py

# Resume from checkpoint
python train.py --resume runs/run_YYYYMMDD_HHMMSS/checkpoints/checkpoint_epoch0010.pt
```

### 6. Inference
```bash
# Single task (by name)
python inference.py \
  --image test.jpg \
  --checkpoint runs/.../best.pt \
  --task "open bottle of beer"

# By task ID
python inference.py --image test.jpg --checkpoint best.pt --task_id 7

# All 14 tasks at once
python inference.py --image test.jpg --checkpoint best.pt --all_tasks

# Arbitrary text (not in predefined 14 tasks)
python inference.py --image test.jpg --checkpoint best.pt --task "pick up the red cup"
```

---

## 3-Phase Training Strategy

| Phase | Epochs | Trainable | LR |
|-------|--------|-----------|-----|
| 1 | 1–10 | FiLM layers only | 1e-3 |
| 2 | 11–50 | YOLOv8n Neck + FiLM | 5e-4 |
| 3 | 51–100 | All layers + FiLM | 1e-4 |

---

## FiLM Hidden Dimensions (Scaled to Feature Channels)

| Neck Layer | Index | Channels | FiLM Hidden | Rationale |
|------------|-------|----------|-------------|-----------|
| P3 (small) | 15 | 64 | 64 | Small objects need precise, lightweight modulation |
| P4 (medium) | 18 | 128 | 128 | Balanced capacity for medium objects |
| P5 (large) | 21 | 256 | 128 | Capped to prevent overfitting on small dataset |

---

## 14 Tasks → COCO Coverage

| Task | COCO Coverage | Custom Data Needed? |
|------|--------------|---------------------|
| step on something | Partial (person) | No |
| sit comfortably | ✅ person+chair+couch | No |
| place flowers | ✅ person+plant+vase | No |
| get potatoes out of fire | ❌ person only | **Yes** |
| water plant | ✅ person+potted plant | No |
| get lemon out of tea | ✅ person+cup | No |
| dig hole | ❌ person only | **Yes** |
| open bottle of beer | ✅ person+bottle | No |
| open parcel | Partial (person) | No |
| serve wine | ✅ person+wine glass | No |
| pour sugar | ✅ person+bowl+spoon | No |
| smear butter | ✅ person+knife | No |
| extinguish fire | ❌ person only | **Yes** |
| pound carpet | ❌ person only | **Yes** |

---

## References
- FiLM: Perez et al., 2018 — https://arxiv.org/abs/1709.07871
- YOLO-World: Cheng et al., 2024 — https://arxiv.org/abs/2401.17270
- YOLOv8: Ultralytics — https://github.com/ultralytics/ultralytics
- Sentence-BERT — https://www.sbert.net/
