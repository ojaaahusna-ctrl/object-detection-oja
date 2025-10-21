import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from keras.layers import TFSMLayer
import torch
from PIL import Image
import io
import os

# -------------------------------
# 📦 Fungsi untuk memuat model ResNet50 (Hugging Face)
# -------------------------------
@st.cache_resource
def load_resnet():
    try:
        st.info("📥 Mengunduh model ResNet50 dari Hugging Face...")
        model_pb = hf_hub_download(
            repo_id="ojahusnaa/resnet50-object-model",
            filename="saved_model.pb",
            repo_type="model"
        )
        model_dir = os.path.dirname(model_pb)

        # Gunakan TFSMLayer karena Keras 3 tidak lagi support SavedModel
        model = TFSMLayer(model_dir, call_endpoint="serving_default")

        st.success("✅ ResNet50 berhasil dimuat dari Hugging Face (TFSMLayer).")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model ResNet50 dari Hugging Face: {e}")
        return None


# -------------------------------
# 📦 Fungsi untuk memuat model YOLO (.pt dari GitHub)
# -------------------------------
@st.cache_resource
def load_yolo():
    try:
        st.info("📦 Memuat model YOLO dari GitHub...")
        model_url = "https://github.com/ojahusnaa/yolo-models/releases/download/v1.0/best.pt"

        # Unduh file YOLO
        yolo_path = "yolo_best.pt"
        if not os.path.exists(yolo_path):
            import requests
            r = requests.get(model_url)
            with open(yolo_path, "wb") as f:
                f.write(r.content)

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_path, force_reload=False)
        st.success("✅ Model YOLO berhasil dimuat dari GitHub.")
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO: {e}")
        return None


# -------------------------------
# 🧠 Fungsi Prediksi
# -------------------------------
def predict_resnet(model, image):
    """Prediksi menggunakan model ResNet50 (TFSMLayer)."""
    try:
        img = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
        preds = model(img_array)
        pred_class = tf.argmax(preds, axis=1).numpy()[0]
        return pred_class
    except Exception as e:
        return f"Error: {e}"


def predict_yolo(model, image):
    """Prediksi menggunakan model YOLO (.pt)."""
    try:
        results = model(image)
        results.render()  # Gambar hasil bounding box
        output_image = Image.fromarray(results.ims[0])
        return output_image
    except Exception as e:
        st.error(f"Error YOLO: {e}")
        return None


# -------------------------------
# 🎨 Tampilan Streamlit
# -------------------------------
st.set_page_config(page_title="Dashboard Prediksi Ganda", layout="wide")
st.title("🧠 Dashboard Prediksi Ganda (ResNet50 + YOLO)")

tab1, tab2 = st.tabs(["📸 Prediksi ResNet50", "🎯 Deteksi YOLO"])

# -------------------------------
# 📸 TAB 1: ResNet50
# -------------------------------
with tab1:
    st.header("Prediksi Klasifikasi Gambar (ResNet50)")
    resnet_model = load_resnet()

    uploaded_img = st.file_uploader("Unggah gambar untuk diklasifikasi", type=["jpg", "jpeg", "png"], key="resnet")
    if uploaded_img is not None and resnet_model is not None:
        img = Image.open(uploaded_img)
        st.image(img, caption="Gambar Input", use_container_width=True)

        if st.button("🔮 Prediksi ResNet50"):
            pred = predict_resnet(resnet_model, img)
            st.success(f"Hasil Prediksi: **{pred}**")

# -------------------------------
# 🎯 TAB 2: YOLO
# -------------------------------
with tab2:
    st.header("Deteksi Objek (YOLO)")
    yolo_model = load_yolo()

    uploaded_img_yolo = st.file_uploader("Unggah gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="yolo")
    if uploaded_img_yolo is not None and yolo_model is not None:
        img = Image.open(uploaded_img_yolo)
        st.image(img, caption="Gambar Input", use_container_width=True)

        if st.button("🚀 Jalankan Deteksi YOLO"):
            output = predict_yolo(yolo_model, img)
            if output is not None:
                st.image(output, caption="Hasil Deteksi", use_container_width=True)
