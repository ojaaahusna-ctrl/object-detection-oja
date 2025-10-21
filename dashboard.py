import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from io import StringIO
from huggingface_hub import hf_hub_download

# ========================== CONFIG ==========================
st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# ========================== PILIH MODEL ==========================
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# ========================== FUNGSI CEK FILE ==========================
def check_and_load_file(path, model_name):
    """Verifikasi path file sebelum load model lokal."""
    if not os.path.exists(path):
        st.error(f"❌ File **{model_name}** tidak ditemukan di path: `{path}`")
        st.info("Pastikan file model YOLO sudah diupload ke folder `model/` di GitHub.")
        return None
    return path

# ==============================================================
# ======================= YOLO (PyTorch) =======================
# ==============================================================

if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model PyTorch - YOLO")

    try:
        @st.cache_resource
        def load_yolo():
            yolo_path = check_and_load_file('model/best.pt', 'best.pt')
            if yolo_path:
                return YOLO(yolo_path)
            return None

        with st.spinner("🔄 Memuat model YOLO..."):
            model = load_yolo()

        if model is not None:
            st.success("✅ Model YOLO berhasil dimuat!")

            with st.sidebar.expander("📊 Info Model YOLO"):
                st.text(str(model.info()))

            st.markdown("### Upload Gambar untuk Deteksi Objek")
            uploaded_file = st.file_uploader(
                "Pilih gambar...", 
                type=['jpg', 'jpeg', 'png'],
                key='yolo'
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📷 Gambar Input")
                    st.image(image, use_column_width=True)

                if st.button("🔍 Deteksi Objek", type="primary"):
                    with st.spinner("Mendeteksi objek..."):
                        results = model(image)

                        with col2:
                            st.subheader("🎯 Hasil Deteksi")
                            result_img = results[0].plot()
                            st.image(result_img, use_column_width=True, channels="BGR")

                        st.markdown("---")
                        st.subheader("📋 Detail Deteksi")

                        boxes = results[0].boxes
                        if len(boxes) > 0:
                            cols = st.columns(3)
                            for i, box in enumerate(boxes):
                                col_idx = i % 3
                                with cols[col_idx]:
                                    st.metric(
                                        f"Objek {i+1}",
                                        model.names[int(box.cls)],
                                        f"{box.conf[0].item():.1%}"
                                    )
                        else:
                            st.info("ℹ Tidak ada objek terdeteksi")

        else:
            st.warning("⚠ Model YOLO belum dimuat karena file tidak ditemukan.")

    except Exception as e:
        st.error(f"❌ Error saat menjalankan YOLO: {e}")

# ==============================================================
# ==================== RESNET50 (TensorFlow) ===================
# ==============================================================

elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50 (Hugging Face)")

    @st.cache_resource
    def load_resnet():
        try:
            model_pb_path = hf_hub_download(
                repo_id="ojahusnaa/resnet50-object-model",
                filename="saved_model.pb",   # pastikan ini sesuai dengan file di Hugging Face
                repo_type="model"
            )
            model_dir = os.path.dirname(model_pb_path)
            model = tf.keras.models.load_model(model_dir)
            st.success("✅ ResNet50 berhasil dimuat dari Hugging Face!")
            return model
        except Exception as e:
            st.error(f"Gagal memuat model ResNet50 dari Hugging Face: {e}")
            return None

    with st.spinner("🔄 Memuat model ResNet50 dari Hugging Face..."):
        model = load_resnet()

    if model is not None:
        with st.sidebar.expander("📊 Arsitektur Model"):
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            st.text(stream.getvalue())

        st.markdown("### Upload Gambar untuk Prediksi")
        uploaded_file = st.file_uploader(
            "Pilih gambar...", 
            type=['jpg', 'jpeg', 'png'],
            key='tf'
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📷 Gambar Input")
                st.image(image, use_column_width=True)

            if st.button("🔮 Prediksi", type="primary"):
                with st.spinner("Melakukan prediksi..."):
                    img_array = np.array(image.resize((224, 224))).astype(np.float32)

                    if img_array.ndim == 2:
                        img_array = np.stack((img_array,)*3, axis=-1)
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]

                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

                    predictions = model.predict(img_array, verbose=0)

                    with col2:
                        st.subheader("🎯 Hasil Prediksi")

                        predicted_index = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_index]

                        CLASSES = ['Cheetah', 'Hyena']  # ganti sesuai dataset kamu
                        predicted_class_name = CLASSES[predicted_index]

                        st.metric("Kelas Prediksi", predicted_class_name)
                        st.metric("Confidence", f"{confidence:.2%}")

                        with st.expander("📊 Probabilitas Lengkap"):
                            for i, prob in enumerate(predictions[0]):
                                st.progress(float(prob), text=f"{CLASSES[i]}: {prob:.4f}")
    else:
        st.warning("⚠ Model ResNet50 belum berhasil dimuat.")

# ==============================================================
# ========================= FOOTER =============================
# ==============================================================

st.markdown("---")
st.markdown("📌 Dibuat oleh **Balqis Isaura** | Powered by Streamlit 🚀")
