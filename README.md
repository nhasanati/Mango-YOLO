# Mango-YOLO

**Evaluating Rare-Class Modeling in YOLOv11n for Imbalanced Agricultural Object Detection: A Mango Case Study**

Nidaul Hasanati, Taufik Djatna, Imas Sukaesih Sitanggang, Arif Imam Suroso — IPB University.

This repository accompanies Paper 2 (JOIV). It studies how adding a rare "Reject" quality class affects a YOLOv11n mango grading detector, comparing a 3-class baseline against a 4-class proposed model.

## Models

Three YOLOv11n runs are provided under `models/`:

| Folder | Classes | AMP | Role |
|---|---|---|---|
| [`models/train2`](models/train2/best.pt) | 4 (Super, Class I, Class II, **Reject**) | disabled | **Proposed model (Model B)** — released `best.pt` |
| [`models/train1`](models/train1/best.pt) | 3 (Super, Class I, Class II) | disabled | Baseline (Model A) |
| [`models/train0`](models/train0/best.pt) | 4 (with Reject) | enabled | AMP-enabled ablation (kept for reproducibility) |

**Released / recommended weights:** [`models/train2/best.pt`](models/train2/best.pt) — the 4-class proposed model. Each checkpoint is 5.3 MB.

## Results

Two evaluation protocols are reported. The single-split numbers describe the released `best.pt` checkpoint; the 5-fold cross-validation numbers describe the robustness of the method.

### Single split (70:30 hold-out)

| Metric | 3-class baseline (`train1`) | 4-class proposed (`train2`) | Δ |
|---|---|---|---|
| **mAP@0.5** | 0.9525 | **0.9558** | +0.0033 |
| mAP@0.5:0.95 | 0.9120 | 0.8838 | −0.0282 |
| Precision | 0.8953 | 0.8889 | −0.0064 |
| Recall | 0.9053 | 0.8744 | −0.0309 |

### 5-fold Cross-Validation (Mean ± Std)

| Metric | 3-class baseline (`train1`) | 4-class proposed (`train2`) | Δ |
|---|---|---|---|
| **mAP@0.5** | 0.8887 ± 0.0479 | **0.8896 ± 0.0581** | +0.0009 |
| mAP@0.5:0.95 | 0.8236 ± 0.0707 | 0.8248 ± 0.0670 | +0.0012 |
| Precision | 0.8669 ± 0.0263 | 0.8623 ± 0.0588 | −0.0046 |
| Recall | 0.7729 ± 0.0997 | 0.8074 ± 0.0797 | +0.0345 |

Adding the rare **Reject** class keeps overall detection quality essentially unchanged (mAP@0.5 +0.0033 single-split, +0.0009 CV) while **improving recall under cross-validation (+0.0345)**, i.e. the model generalizes better without sacrificing accuracy on the majority classes.

### Per-class AP — 4-class proposed model (single split)

| Metric | Extra/Super | Class I | Class II | Reject | Mean |
|---|---|---|---|---|---|
| mAP@0.5 | 0.9624 | 0.9517 | 0.9333 | 0.9757 | 0.9558 |
| mAP@0.5:0.95 | 0.9538 | 0.8672 | 0.9228 | 0.7916 | 0.8838 |
| Precision | 0.8444 | 0.9024 | 0.8088 | 1.0000 | 0.8889 |
| Recall | 0.9500 | 0.8721 | 0.8651 | 0.8102 | 0.8744 |

## Training configuration

| Parameter | Value |
|---|---|
| Base model | `yolo11n.pt` (YOLO11 Nano) |
| Task | Object Detection |
| Epochs | 100 |
| Batch size | 4 |
| Image size | 480 × 480 |
| Optimizer | Auto (SGD / AdamW) |
| Initial LR (lr0) | 0.01 |
| Early-stopping patience | 100 |
| Pretrained weights | Yes |
| AMP (mixed precision) | Disabled (`train0` uses enabled, for ablation) |
| IoU threshold | 0.7 |
| Device | NVIDIA MX450 |

## Usage

```python
from ultralytics import YOLO

# Load the proposed 4-class model
model = YOLO("models/train2/best.pt")

# Run inference on an image
results = model("path/to/mango.jpg")
results[0].show()
```

## Repository structure

```
models/train2/best.pt   # 4-class proposed model (recommended)
models/train1/best.pt   # 3-class baseline
models/train0/best.pt   # 4-class, AMP enabled (ablation)
vqa/                     # Visual Question Answering (multi-answer grading)
data/                    # Datasets
```
