from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
from collections import Counter

st.set_page_config(page_title="Deteksi Mangga", layout="centered")
st.title("Upload & Deteksi Gambar Mangga")

# ==========================================================
# Fixes over the old version:
# 1. Add `from ultralytics import YOLO` (previously missing -> NameError).
# 2. conf / iou / imgsz are NO longer hardcoded -> controlled via sidebar sliders.
# 3. Drop the manual NMS (cv2.dnn.NMSBoxes). Ultralytics already runs NMS
#    itself through the `iou` argument in model.predict(), so a manual NMS is
#    redundant and makes the results inconsistent.
# 4. imgsz default 480 -> MUST match the training image size (model trained at 480).
# 5. default_conf = 0.30 and default_iou = 0.55 (precision comparable to 0.40, but
#    faint Reject detections are still captured, given the small amount of Reject
#    data in training).
# ==========================================================

MODEL_PATH = "models/train2/best.pt"                 # 4 classes (proposed model)
TRAIN_IMGSZ = 480                                     # match args.yaml from training
default_conf = 0.30
default_iou = 0.55


@st.cache_resource
def load_model(path):
    return YOLO(path)


# ---------- Threshold settings panel ----------
st.sidebar.header("⚙️ Setting Threshold")

conf_thres = st.sidebar.slider(
    "Confidence (conf)", 0.05, 0.90, default_conf, 0.05,
    help="Default 0.30: Reject samar (mis. reject_6 conf 0.34) ikut tertangkap. "
         "Naik = lebih ketat (buang objek palsu). Turun = lebih sensitif."
)
nms_iou = st.sidebar.slider(
    "NMS IoU (iou)", 0.30, 0.90, default_iou, 0.05,
    help="Turun = lebih agresif buang box yang tumpang tindih."
)
imgsz = st.sidebar.select_slider(
    "Image size (imgsz)", options=[480, 640, 800, 1024], value=TRAIN_IMGSZ,
    help="Disarankan 480 (sama dengan training). Nilai lain hanya untuk eksperimen."
)
agnostic = st.sidebar.checkbox(
    "Agnostic NMS (buang box dobel lintas-kelas)", value=True,
    help="Berguna kalau 1 mangga kedeteksi 2 kelas sekaligus (misal Class 1 + Reject)."
)

if imgsz != TRAIN_IMGSZ:
    st.sidebar.warning(
        f"imgsz={imgsz} beda dengan training ({TRAIN_IMGSZ}). "
        "Hasil bisa aneh — pakai 480 kalau ragu."
    )

model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Gambar gagal dibaca. Coba upload ulang.")
        st.stop()

    # show the original image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Gambar yang diupload", use_container_width=True)

    # ==========================================================
    # Prediction: conf + iou + imgsz taken directly from the sliders. NO manual NMS.
    # ==========================================================
    results = model.predict(
        img_bgr,
        conf=conf_thres,
        iou=nms_iou,
        imgsz=imgsz,
        max_det=300,
        agnostic_nms=agnostic,
        verbose=False,
    )
    r0 = results[0]

    # plot the results
    result_bgr = r0.plot()
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption="Hasil Deteksi", use_container_width=True)

    # ===== COUNT OBJECT =====
    if r0.boxes is None or len(r0.boxes) == 0:
        st.warning("Tidak ada objek terdeteksi. Coba turunkan Confidence di sidebar.")
    else:
        cls_ids = r0.boxes.cls.cpu().numpy().astype(int).tolist()
        confs = r0.boxes.conf.cpu().numpy().tolist()
        names = model.names

        labels = [names[i] for i in cls_ids]
        counts = Counter(labels)

        st.subheader("Count Objek")
        st.metric("Total objek terdeteksi", sum(counts.values()))

        st.write("**Mangga yang terdeteksi:**")
        unique_classes = list(dict.fromkeys(labels))
        st.write(", ".join(unique_classes))

        st.write("**Jumlah objek berdasarkan kelas:**")
        for label, count in counts.items():
            st.write(f"- {label}: {count}")

        # per-object confidence (to inspect weak detections near the threshold)
        with st.expander("Detail confidence tiap objek"):
            for i, (lbl, c) in enumerate(zip(labels, confs), start=1):
                st.write(f"{i}. {lbl} — conf {c:.2f}")

    st.success("Deteksi selesai!")
