# Mango-YOLO

Evaluating Rare-Class Modeling in YOLOv11n for Imbalanced Agricultural Object Detection: A Mango Case Study.

## Model

The final detection model is a **YOLOv11n** network trained for mango detection and quality grading.

| Item | Detail |
|---|---|
| Architecture | YOLOv11n (nano) |
| Training run | `train16` |
| Weights | [`models/train16/best.pt`](models/train16/best.pt) |
| Size | 5.3 MB |
| **mAP@0.5** | **0.931** |

### Training note

`train16` was trained with **Automatic Mixed Precision disabled (`amp=False`)**. This was the key change over the earlier `train12` run (`amp=True`), which suffered from `NaN` validation loss and reached only mAP@0.5 = 0.773. Disabling AMP stabilized training and raised mAP@0.5 from 0.773 to 0.931, which is why `train16` is used as the final model.

## Usage

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("models/train16/best.pt")

# Run inference on an image
results = model("path/to/mango.jpg")
results[0].show()
```

## Repository structure

```
models/train16/best.pt   # Final YOLOv11n weights (train16)
vqa/                      # Visual Question Answering (multi-answer grading)
data/                     # Datasets
```
