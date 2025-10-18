import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2

# --- Konfigurasi ---
# GANTI INI dengan label kelas Anda yang sebenarnya
CLASS_NAMES = ["Kardus", "Kertas", "Plastik", "Kaleng", "Lain-lain"] 
TARGET_SIZE = (224, 224) # Ukuran input model klasifikasi

# ==========================
# Load Models (Menggunakan st.cache_resource)
# ==========================
@st.cache_resource
def load_models():
    # Inisialisasi variabel di luar blok try untuk memastikan scope yang jelas
    yolo_model = None
    classifier = None

    # --- Pemuatan Model YOLO ---
    try:
        st.info("Memuat model YOLO (Deteksi Objek)...")
        yolo_model = YOLO("model/best.pt")
        st.success("Model YOLO berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO (best.pt). Pastikan file ada di folder 'model'. Detail error: {e}")

    # --- Pemuatan Model Klasifikasi ---
    try:
        st.info("Memuat model Klasifikasi...")
        # Menggunakan compile=False untuk mengatasi masalah ValueError/dtype akibat perbedaan versi TF
        classifier = tf.keras.models.load_model(
            "model/Raudhatul Husna_laporan2.h5", 
            custom_objects=None,
            compile=False 
        )
        st.success("Model Klasifikasi berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model Klasifikasi. Pastikan file 'Raudhatul Husna_laporan2.h5' ada di folder 'model'. Detail error: {e}")

    return yolo_model, classifier

# Panggil fungsi load_models() sekali di awal skrip
yolo_model, classifier = load_models()

# ==========================
# UI & Main Logic
# ==========================
st.title("🧠 Image Classification & Object Detection App")
st.caption("Aplikasi ini menggunakan Model YOLO untuk Deteksi Objek dan Model Keras untuk Klasifikasi Gambar.")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # ==========================
    # Deteksi Objek (YOLO)
    # ==========================
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is not None:
            st.subheader("🎯 Hasil Deteksi Objek")
            try:
                # Proses deteksi
                results = yolo_model(img)
                
                # Mengambil gambar hasil deteksi
                result_img = results[0].plot() 
                
                # Tampilkan hasil
                st.image(result_img, caption="Hasil Deteksi Objek", use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi objek: {e}")
        else:
            st.warning("⚠️ Model YOLO tidak dimuat. Tidak dapat menjalankan deteksi objek.")

    # ==========================
    # Klasifikasi Gambar
    # ==========================
    elif menu == "Klasifikasi Gambar":
        if classifier is not None:
            st.subheader("📊 Hasil Klasifikasi")
            try:
                # 1. Preprocessing
                img_resized = img.resize(TARGET_SIZE)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) # Tambah dimensi batch
                img_array = img_array / 255.0 # Normalisasi

                # 2. Prediksi
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # 3. Tampilkan Hasil
                predicted_class = CLASS_NAMES[class_index]
                
                st.success(f"✅ Klasifikasi: **{predicted_class}**")
                st.metric("Tingkat Keyakinan", f"{confidence * 100:.2f} %")
                
                st.write("---")
                st.write("Detail Probabilitas:")
                st.json({name: f"{prob * 100:.2f}%" for name, prob in zip(CLASS_NAMES, prediction[0])})

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
        else:
            st.warning("⚠️ Model Klasifikasi tidak dimuat. Tidak dapat menjalankan klasifikasi gambar.")
