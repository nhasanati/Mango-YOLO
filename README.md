# Mango-YOLO

**Evaluating Rare-Class Modeling in YOLOv11n for Imbalanced Agricultural Object Detection: A Mango Case Study**

Nidaul Hasanati, Taufik Djatna, Imas Sukaesih Sitanggang, Arif Imam Suroso — IPB University, UIN Syarif Hidayatullah

This repository accompanies the paper titled above. It studies how adding a rare **Reject** quality class affects a YOLOv11n mango grading detector, comparing a 3-class baseline (Super/Extra Class, Class I, Class II) against a 4-class proposed model (Super/Extra Class, Class I, Class II, **Reject**).

A ready-to-run Streamlit demo ([`main.py`](main.py)), built on the 4-class model ([`models/train2`](models/train2/best.pt)), is included for interactive detection on your own images — see [Streamlit demo app](#streamlit-demo-app).

## Models

Three YOLOv11n runs are provided under `models/`:

| Folder | Classes | AMP | Role |
|---|---|---|---|
| [`models/train2`](models/train2/best.pt) | 4 (Extra Class, Class I, Class II, **Reject**) | disabled | **Proposed model (Model B)** — released `best.pt` |
| [`models/train1`](models/train1/best.pt) | 3 (Extra Class, Class I, Class II) | disabled | Baseline (Model A) |
| [`models/train0`](models/train0/best.pt) | 4 (Extra Class, Class I, Class II, Reject) | enabled | AMP-enabled ablation (kept for reproducibility) |

**Released / recommended weights:** [`models/train2/best.pt`](models/train2/best.pt) — the 4-class proposed model. Each checkpoint is 5.3 MB.

## Dataset

Both datasets are released under `data/` in YOLO format (one `.txt` label file per image, `class x_center y_center width height` normalised to 0–1), one folder per configuration. Class indices follow the order in each `data.yaml` (`0 Class 1`, `1 Class 2`, `2 Extra Class`, `3 Reject`).

### Image sources

The 468 post-harvest mango images come from two sources:

1. **Mango Varieties Classification and Grading** (Kaggle, S. Shahane, 2021) — single-mango images captured under controlled studio conditions, originally labelled with three standardized quality classes (Extra Class, Class I, Class II). https://www.kaggle.com/datasets/saurabhshahane/mango-varieties-classification
2. **Additional real-world images** collected from publicly available online sources — multi-mango scenes of different varieties and quality levels, including severely defective (Reject) fruit, added to introduce environmental variability and represent practical sorting conditions.

The Kaggle images remain subject to the licence stated on their Kaggle page. The bounding-box annotations in this repository were created for the paper and are released together with the images for research use.

### Classes (SNI 3164:2024)

Labels follow the Indonesian national mango standard **SNI 3164:2024**, which grades fruit by the share of the skin surface affected by minor defects (scratches, sunburn, sap stains, abrasions that do not reach the flesh). The proposed configuration adds a fourth, non-commercial class.

| Index | Class | Definition | Annotations (share) |
|---|---|---|---|
| 2 | **Extra Class** (Super Class) | highest quality, no significant defects | 187 (25.1%) |
| 0 | **Class I** | commercially acceptable, minor skin defects ≤ 5% of surface | 349 (46.8%) |
| 1 | **Class II** | satisfies general requirements, minor skin defects ≤ 10% of surface | 159 (21.3%) |
| 3 | **Reject** (4-class only) | severe damage, disease symptoms, deformities, or defects reaching the flesh — unsuitable for commercial distribution | 50 (6.7%) |

Reject is deliberately a **rare class**: 50 of 745 annotations (6.7%), 30 of them in the training set. This imbalance is the object of study, not an artefact to be corrected.

### Annotation protocol

All images were annotated with bounding boxes in YOLO format by **two annotators working independently**, following a written guideline derived from SNI 3164:2024, the original Kaggle annotation protocol, and a decision flow (Figure 2 of the paper). Class assignment used the percentage-based defect thresholds above. Borderline cases at the Class II–Reject boundary, where the decision depends on the severity and extent of damage, received particular attention; disagreements were resolved by discussion until consensus was reached.

### Partitions

| | 3-class baseline (`data/3-class`) | 4-class proposed (`data/4-class`) |
|---|---|---|
| Total images / annotations | 448 / 695 | 468 / 745 |
| `images/train` | 312 / 525 | 327 / 555 |
| `images/test` — 30% evaluation partition | 136 / 170 | 141 / 190 |

- **Split.** A single 70:30 partition is used in both configurations, exactly as in the paper. The 3-class dataset is derived from the 4-class one by removing the 20 images that contain Reject fruit (15 from train, 5 from the evaluation partition), so `3-class/images/test` is a subset of `4-class/images/test`.
- **What `val:` means.** Each `data.yaml` points `val:` at `images/test`, which is the configuration actually used for training: the best checkpoint (`best.pt`) was selected on this 30% partition, and the single-split numbers below are measured on the same partition. The comparison between the two models is fair because the protocol is identical for both, and the 5-fold cross-validation is the primary evidence for the conclusions.
- **Duplicate check.** MD5 hashes of all 468 images were compared across partitions. One evaluation image (`4-class/images/test/376c8fb7-reject_10.jpg`) is byte-identical to a training image (`4-class/images/train/5827c6a2-reject_5.jpg`). It is kept so the released data matches what was trained on; excluding it changes mAP@0.5 of the 4-class model by less than 0.001. The 3-class dataset contains no duplicates.
- **Relation to MangoVQA.** The follow-up Visual Question Answering work ([MangoVQA](https://github.com/nhasanati/MangoVQA)) further divides this 141-image evaluation partition into 70 validation and 71 test images. That finer split belongs to the VQA project and is documented there; the detection results in this repository use the full 141-image partition.

To reproduce the single-split numbers from the released weights (run from the repository root):

```bash
yolo val model=models/train2/best.pt data=data/4-class/data.yaml imgsz=480   # 4-class proposed
yolo val model=models/train1/best.pt data=data/3-class/data.yaml imgsz=480   # 3-class baseline
```

Small differences (≈0.003–0.004 in mAP@0.5) can arise from the Ultralytics version.

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
| Precision | 0.8444 | 0.9024 | 0.8088 | **1.0000** | 0.8889 |
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
data/3-class/            # 3-class baseline dataset (train 312 / test 136)
data/4-class/            # 4-class proposed dataset (train 327 / test 141)
```
