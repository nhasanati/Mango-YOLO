# Mango-YOLO

**Evaluating Rare-Class Modeling in YOLOv11n for Imbalanced Agricultural Object Detection: A Mango Case Study**

Nidaul Hasanati, Taufik Djatna, Imas Sukaesih Sitanggang, Arif Imam Suroso — IPB University.

This repository accompanies the paper titled above. It studies how adding a rare **Reject** quality class affects a YOLOv11n mango grading detector, comparing a 3-class baseline against a 4-class proposed model.

A ready-to-run Streamlit demo ([`main.py`](main.py)) is included for interactive detection on your own images — see [Streamlit demo app](#streamlit-demo-app).

## Models

Three YOLOv11n runs are provided under `models/`:

| Folder | Classes | AMP | Role |
|---|---|---|---|
| [`models/train2`](models/train2/best.pt) | 4 (Super/Extra Class, Class I, Class II, **Reject**) | disabled | **Proposed model (Model B)** — released `best.pt` |
| [`models/train1`](models/train1/best.pt) | 3 (Super/Extra Class, Class I, Class II) | disabled | Baseline (Model A) |
| [`models/train0`](models/train0/best.pt) | 4 (Super/Extra Class, Class I, Class II, Reject) | enabled | AMP-enabled ablation (kept for reproducibility) |

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

### Ablation — Automatic Mixed Precision (AMP)

`train0` and `train2` use the identical 4-class setup; the only difference is AMP. The metrics below are each run's **own validation at the final epoch (100)** — not the 70:30 test set used in the tables above — so compare them only against each other.

| Metric | `train0` (AMP on) | `train2` (AMP off) |
|---|---|---|
| mAP@0.5 | 0.7732 | **0.9309** |
| mAP@0.5:0.95 | 0.7390 | **0.8590** |
| Precision | 0.8487 | 0.8370 |
| Recall | 0.7333 | **0.8803** |
| Validation loss | **NaN** (unstable) | normal |

AMP produced NaN validation loss and unstable training on this dataset; disabling it stabilized training and raised mAP@0.5 by ~0.16. This is why the released model (`train2`) is trained with AMP disabled.

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

### Streamlit demo app

`main.py` is an interactive Streamlit app for uploading an image and running mango
detection with adjustable confidence / IoU / image-size thresholds. The model weights
are already included in the repository, so after cloning you only need to install the
dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/nhasanati/Mango-YOLO.git
cd Mango-YOLO

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app (must be run from the repo root)
streamlit run main.py
```

The app then opens in your browser at http://localhost:8501 and loads
`models/train2/best.pt` (the 4-class proposed model) by default.

> **Note:** PyTorch is installed automatically as an `ultralytics` dependency (CPU build). For GPU acceleration, install the matching CUDA build of `torch` first — see [pytorch.org](https://pytorch.org).

## Repository structure

```
main.py                 # Streamlit detection demo app
models/train2/best.pt   # 4-class proposed model (recommended)
models/train1/best.pt   # 3-class baseline
models/train0/best.pt   # 4-class, AMP enabled (ablation)
vqa/                     # Visual Question Answering (multi-answer grading)
data/                    # Datasets
```
