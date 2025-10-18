import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- Konfigurasi ---
# Catatan: Kita menggunakan model YOLO untuk kedua mode. 
# GANTI INI dengan label kelas yang BENAR dari model YOLO Anda
# Contoh labels untuk deteksi:
YOLO_CLASS_NAMES = {
    0: "Kardus", 
    1: "Kertas", 
    2: "Plastik", 
    3: "Kaleng", 
    4: "Lain-lain"
} 
# Tentukan kelas mana yang paling mungkin muncul jika gambar didominasi oleh satu objek.
CLASSIFICATION_THRESHOLD = 0.5 # Rasio deteksi minimum untuk dianggap sebagai klasifikasi utama

# ===============================================
# Load Model (Hanya YOLO)
# ===============================================
@st.cache_resource
def load_yolo_model():
    yolo_model = None
    try:
        st.info("Memuat model YOLO (best.pt)...")
        yolo_model = YOLO("model/best.pt") 
        st.success("Model YOLO berhasil dimuat. Siap untuk Deteksi & Klasifikasi.")
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLO. Detail error: {e}")
    return yolo_model

# Panggil fungsi load_models()
yolo_model = load_yolo_model()

# ===============================================
# Logika Klasifikasi Berbasis Deteksi
# ===============================================
def classify_by_detection(results):
    """Menentukan klasifikasi gambar berdasarkan objek yang paling sering terdeteksi."""
    if not results or not results[0].boxes:
        return "Tidak Terdeteksi", 0.0

    boxes = results[0].boxes
    total_area = 0
    class_areas = {}

    # 1. Hitung total area bounding box dan area per kelas
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0].item())
        area = (x2 - x1) * (y2 - y1)
        total_area += area
        
        class_areas[cls] = class_areas.get(cls, 0) + area

    if total_area == 0:
        return "Tidak Terdeteksi", 0.0

    # 2. Cari kelas dengan kontribusi area terbesar
    max_area = 0
    best_cls_id = -1
    for cls_id, area in class_areas.items():
        if area > max_area:
            max_area = area
            best_cls_id = cls_id

    # 3. Hitung rasio
    confidence_ratio = max_area / sum(class_areas.values()) if sum(class_areas.values()) > 0 else 0.0
    
    if best_cls_id in YOLO_CLASS_NAMES:
        return YOLO_CLASS_NAMES[best_cls_id], confidence_ratio
    else:
        return "Lain-lain", confidence_ratio


# ===============================================
# UI & Main Logic
# ===============================================
st.title("🧠 Image Classification & Object Detection App")
st.caption("Aplikasi menggunakan Model YOLO Anda untuk Deteksi Objek dan Klasifikasi Gambar.")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if yolo_model is None:
        st.error("Tidak ada model yang dimuat. Silakan periksa error pemuatan YOLO.")
    
    else:
        # ==========================
        # Deteksi Objek (YOLO)
        # ==========================
        if menu == "Deteksi Objek (YOLO)":
            st.subheader("🎯 Hasil Deteksi Objek")
            try:
                # Proses deteksi
                results = yolo_model(img)
                result_img = results[0].plot() 
                st.image(result_img, caption="Hasil Deteksi Objek", use_container_width=True)
            except Exception as e:
                st.error(f"Terjadi kesalahan saat deteksi objek: {e}")

        # ==========================
        # Klasifikasi Gambar (Berbasis YOLO)
        # ==========================
        elif menu == "Klasifikasi Gambar":
            st.subheader("📊 Hasil Klasifikasi Gambar (Menggunakan Model YOLO)")
            try:
                # 1. Jalankan deteksi
                results = yolo_model(img)
                
                # 2. Klasifikasi hasil deteksi
                predicted_class, confidence = classify_by_detection(results)

                # 3. Tampilkan Hasil
                st.success(f"✅ Klasifikasi Gambar: **{predicted_class}**")
                st.metric("Rasio Dominasi Objek", f"{confidence * 100:.2f} %")
                
                if confidence < CLASSIFICATION_THRESHOLD and predicted_class != "Tidak Terdeteksi":
                    st.warning(f"Rasio dominasi objek ({confidence:.2f}) di bawah batas klasifikasi ({CLASSIFICATION_THRESHOLD}).")

                # (Opsional) Tampilkan hasil deteksi sebagai referensi
                st.write("---")
                st.write("Visualisasi Deteksi Objek sebagai Referensi:")
                result_img = results[0].plot()
                st.image(result_img, use_container_width=True)


            except Exception as e:
                st.error(f"Terjadi kesalahan saat klasifikasi berbasis deteksi: {e}")
