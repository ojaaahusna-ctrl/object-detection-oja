import streamlit as st
import torch
from ultralytics import YOLO
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image
import numpy as np
import io
import os

# ======================================
# 🔧 CONFIG
# ======================================
st.set_page_config(page_title="Object Detection & Classification Dashboard", page_icon="📸", layout="wide")

st.title("📸 Object Detection & Classification Dashboard")
st.markdown("Gunakan **YOLO (best.pt)** dan **ResNet50** bersama-sama.")
st.divider()

# ======================================
# 🧩 Fungsi untuk memuat YOLO
# ======================================
@st.cache_resource
def load_yolo_model():
    model_path = "model/best.pt"
    if not os.path.exists(model_path):
        st.error("⚠️ File YOLO model (best.pt) tidak ditemukan di folder 'model/'.")
        return None
    model = YOLO(model_path)
    return model

# ======================================
# 🧩 Fungsi untuk memuat ResNet50 dari Hugging Face
# ======================================
@st.cache_resource
def load_resnet50_from_hf():
    MODEL_ID = "ojahusnaa/resnet50-object-model"
    try:
        local_dir = hf_hub_download(repo_id=MODEL_ID, filename="ResNet50_Universal", repo_type="model")
        model = tf.keras.models.load_model(local_dir)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model ResNet50 dari Hugging Face: {e}")
        return None

# ======================================
# 🧩 Fungsi preprocessing ResNet50
# ======================================
def preprocess_resnet_image(image):
    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    return img_array

# ======================================
# 🧩 Fungsi prediksi ResNet50
# ======================================
def predict_resnet(model, image):
    img_array = preprocess_resnet_image(image)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    return class_idx, confidence

# ======================================
# 📤 Upload Gambar
# ======================================
st.subheader("Unggah gambar untuk prediksi")
uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    st.subheader("🔍 Hasil Prediksi")

    # Load kedua model
    yolo_model = load_yolo_model()
    resnet_model = load_resnet50_from_hf()

    if yolo_model is None or resnet_model is None:
        st.stop()

    # YOLO Prediction
    st.info("🚀 Menjalankan deteksi YOLO...")
    results = yolo_model.predict(image, conf=0.4, imgsz=640)
    yolo_img = results[0].plot()  # hasil bounding box
    st.image(yolo_img, caption="Hasil Deteksi YOLO", use_container_width=True)

    # ResNet50 Prediction
    st.info("🧠 Menjalankan klasifikasi ResNet50...")
    class_idx, confidence = predict_resnet(resnet_model, image)
    st.success(f"✅ Kelas Prediksi ResNet50: {class_idx} (Confidence: {confidence:.2f})")

else:
    st.warning("📂 Silakan unggah gambar terlebih dahulu untuk melakukan prediksi.")
