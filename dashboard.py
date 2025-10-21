import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
from huggingface_hub import hf_hub_download
from io import StringIO
import os

# ==================== KONFIGURASI DASAR ====================
st.set_page_config(
    page_title="Dashboard Klasifikasi & Deteksi Gambar",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Klasifikasi & Deteksi Gambar - Balqis Isaura")
st.markdown("---")

# Sidebar pilihan model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# ==================== MODEL YOLO ====================
if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model Deteksi Objek - YOLO (PyTorch)")

    @st.cache_resource
    def load_yolo():
        path = "model/best.pt"
        if not os.path.exists(path):
            st.error(f"❌ File model YOLO tidak ditemukan di `{path}`")
            return None
        return YOLO(path)

    with st.spinner("Memuat model YOLO..."):
        yolo_model = load_yolo()

    if yolo_model:
        st.success("✅ Model YOLO berhasil dimuat!")

        # Upload gambar
        uploaded_file = st.file_uploader("Pilih gambar untuk deteksi objek", type=["jpg", "jpeg", "png"], key="yolo")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔍 Jalankan Deteksi", type="primary"):
                with st.spinner("Sedang mendeteksi objek..."):
                    results = yolo_model(image)

                    with col2:
                        st.subheader("🎯 Hasil Deteksi")
                        result_img = results[0].plot()
                        st.image(result_img, use_column_width=True, channels="BGR")

                    # Tampilkan detail deteksi
                    boxes = results[0].boxes
                    st.markdown("---")
                    st.subheader("📋 Detail Deteksi")

                    if len(boxes) > 0:
                        for i, box in enumerate(boxes):
                            cls_name = yolo_model.names[int(box.cls)]
                            conf = box.conf[0].item()
                            st.metric(f"Objek {i+1}", cls_name, f"{conf:.1%}")
                    else:
                        st.info("ℹ Tidak ada objek terdeteksi.")
    else:
        st.warning("⚠ Tidak dapat memuat model YOLO.")

# ==================== MODEL RESNET50 ====================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model Klasifikasi Gambar - ResNet50 (TensorFlow)")

    MODEL_ID = "ojahusnaa/resnet50-object-model"  # dari Hugging Face

    @st.cache_resource
    def load_resnet_from_hf():
        try:
            st.info("📦 Mengunduh model dari Hugging Face Hub...")
            local_path = hf_hub_download(
                repo_id=MODEL_ID,
                filename="ResNet50_Universal",
                repo_type="model"
            )
            model = tf.keras.models.load_model(local_path)
            st.success("✅ Model ResNet50 berhasil dimuat dari Hugging Face Hub!")
            return model
        except Exception as e:
            st.error(f"Gagal memuat model ResNet50: {e}")
            return None

    with st.spinner("Memuat model ResNet50..."):
        resnet_model = load_resnet_from_hf()

    if resnet_model:
        with st.sidebar.expander("📊 Arsitektur Model"):
            stream = StringIO()
            resnet_model.summary(print_fn=lambda x: stream.write(x + "\n"))
            st.text(stream.getvalue())

        # Upload gambar
        uploaded_file = st.file_uploader("Pilih gambar untuk klasifikasi", type=["jpg", "jpeg", "png"], key="resnet")

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔮 Prediksi Kelas", type="primary"):
                with st.spinner("Melakukan klasifikasi..."):
                    img_array = np.array(image.resize((224, 224))).astype(np.float32)

                    # Pastikan channel RGB
                    if img_array.ndim == 2:
                        img_array = np.stack((img_array,) * 3, axis=-1)
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                    # Prediksi
                    predictions = resnet_model.predict(img_array, verbose=0)
                    predicted_index = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_index]

                    # Sesuaikan label kelas sesuai model kamu
                    CLASSES = ["Cheetah", "Hyena"]
                    predicted_label = CLASSES[predicted_index]

                    with col2:
                        st.subheader("🎯 Hasil Klasifikasi")
                        st.metric("Kelas Prediksi", predicted_label)
                        st.metric("Confidence", f"{confidence:.2%}")

                    with st.expander("📊 Semua Probabilitas"):
                        for i, prob in enumerate(predictions[0]):
                            st.progress(float(prob), text=f"{CLASSES[i]}: {prob:.4f}")
    else:
        st.warning("⚠ Tidak dapat memuat model ResNet50.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("📌 Dibuat oleh **Balqis Isaura** | Powered by Streamlit 🚀")
