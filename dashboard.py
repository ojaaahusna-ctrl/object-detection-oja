import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
import os

# --- Konfigurasi ---
CLASS_NAMES = ["Kardus", "Kertas", "Plastik", "Kaleng", "Lain-lain"] 
TARGET_SIZE = (224, 224) 

# ===============================================
# Load Models
# ===============================================
@st.cache_resource
def load_models():
    yolo_model = None
    classifier = None

    # --- Pemuatan Model YOLO ---
    try:
        st.info("Memuat model YOLO (Deteksi Objek)...")
        yolo_model = YOLO("model/best.pt") 
        st.success("Model YOLO berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO. Detail error: {e}")

    # --- Pemuatan Model Klasifikasi (Menggunakan TFSMLayer) ---
    try:
        st.info("Memuat model Klasifikasi (Keras 3 / SavedModel)...")
        
        # SOLUSI FINAL untuk 'File format not supported' di Keras 3:
        # Menggunakan TFSMLayer untuk memuat SavedModel lama.
        classifier = tf.keras.layers.TFSMLayer(
            "model/klasifikasi_saved_model", 
            call_endpoint='serving_default' # Endpoint standar SavedModel
        )
        st.success("Model Klasifikasi berhasil dimuat.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model Klasifikasi. Detail error: {e}")

    return yolo_model, classifier

# Panggil fungsi load_models()
yolo_model, classifier = load_models()

# ===============================================
# UI & Main Logic
# ===============================================
st.title("🧠 Image Classification & Object Detection App")
st.caption("Aplikasi ini menggunakan Model YOLO untuk Deteksi Objek dan Model Keras untuk Klasifikasi Gambar.")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    # Deteksi Objek
    if menu == "Deteksi Objek (YOLO)":
        if yolo_model is not None:
            st.subheader("🎯 Hasil Deteksi Objek")
            try:
                results = yolo_model(img)
                result_img = results[0].plot() 
                st.image(result_img, caption="Hasil Deteksi Objek", use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi objek: {e}")
        else:
            st.warning("⚠️ Model YOLO tidak dimuat.")

    # Klasifikasi Gambar
    elif menu == "Klasifikasi Gambar":
        if classifier is not None:
            st.subheader("📊 Hasil Klasifikasi")
            try:
                # 1. Preprocessing
                img_resized = img.resize(TARGET_SIZE)
                img_array = image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) 
                img_array = img_array / 255.0 

                # 2. Prediksi menggunakan TFSMLayer
                # Input harus berupa dict (format yang dibutuhkan SavedModel)
                input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
                
                # Panggil layer
                prediction_output = classifier(input_tensor)
                
                # Ekstraksi tensor dari output (diasumsikan outputnya adalah dict dengan satu tensor)
                prediction_tensor = prediction_output[list(prediction_output.keys())[0]]
                
                # Konversi ke NumPy
                prediction = prediction_tensor.numpy()
                
                # 3. Hasil
                class_index = np.argmax(prediction)
                confidence = np.max(prediction)
                predicted_class = CLASS_NAMES[class_index]
                
                st.success(f"✅ Klasifikasi: **{predicted_class}**")
                st.metric("Tingkat Keyakinan", f"{confidence * 100:.2f} %")
                
                st.write("---")
                st.write("Detail Probabilitas:")
                st.json({name: f"{prob * 100:.2f}%" for name, prob in zip(CLASS_NAMES, prediction[0])})

            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi: {e}")
                st.write("Pastikan SavedModel Anda memiliki output tensor yang benar.")
        else:
            st.warning("⚠️ Model Klasifikasi tidak dimuat.")
