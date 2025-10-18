import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image
import numpy as np
import os # Tambah os untuk path checking
from io import StringIO # Untuk model summary

st.set_page_config(
    page_title="Dashboard Model - Balqis Isaura",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Dashboard Model - Balqis Isaura")
st.markdown("---")

# Sidebar untuk pilih model
model_choice = st.sidebar.radio(
    "Pilih Model:",
    ["PyTorch - YOLO", "TensorFlow - ResNet50"]
)

# Definisikan nama file H5 (PILIH SALAH SATU YANG BENAR DI REPO ANDA)
TF_MODEL_FILENAME = 'Raudhatul Husna_laporan2.h5' # <--- GANTI JIKA NAMA FILE DI REPO ANDA BERBEDA!

# ==================== FUNGSI LOAD & CHECK MODEL ====================

def check_and_load_file(path, model_name):
    """Membantu memverifikasi path file sebelum memuat model."""
    if not os.path.exists(path):
        st.error(f"❌ File **{model_name}** tidak ditemukan di path: `{path}`")
        st.info("Pastikan file model sudah di-commit dan di-push ke folder `model/`")
        return None
    return path

# ==================== MODEL PYTORCH YOLO ====================
if model_choice == "PyTorch - YOLO":
    st.header("🎯 Model PyTorch - YOLO")
    
    try:
        # Load YOLO model
        @st.cache_resource
        def load_yolo():
            yolo_path = check_and_load_file('model/best.pt', 'best.pt')
            if yolo_path:
                return YOLO(yolo_path)
            return None
        
        with st.spinner("Loading YOLO model..."):
            model = load_yolo()
        
        if model is not None:
            st.success("✅ Model YOLO berhasil dimuat!")
            
            # Info model di sidebar
            with st.sidebar.expander("📊 Info Model"):
                st.text(str(model.info()))
            
            # Upload gambar
            st.markdown("### Upload Gambar untuk Deteksi Objek")
            uploaded_file = st.file_uploader(
                "Pilih gambar...", 
                type=['jpg', 'jpeg', 'png'],
                key='yolo'
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB") # Pastikan konversi ke RGB
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📷 Gambar Input")
                    st.image(image, use_column_width=True)
                
                if st.button("🔍 Deteksi Objek", type="primary"):
                    with st.spinner("Mendeteksi objek..."):
                        # Prediksi
                        results = model(image)
                        
                        with col2:
                            st.subheader("🎯 Hasil Deteksi")
                            result_img = results[0].plot()
                            st.image(result_img, use_column_width=True, channels="BGR") # YOLO plot output BGR, Streamlit butuh channels="BGR"
                        
                        # Detail deteksi
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
                                        f"{box.conf[0].item():.1%}" # Gunakan .item() untuk tensor
                                    )
                        else:
                            st.info("ℹ Tidak ada objek terdeteksi")
                        
        else: # Model tidak dimuat karena FileNotFoundError
             st.info("⚠ Tidak dapat melanjutkan karena model YOLO tidak ditemukan.")

    except Exception as e:
        st.error(f"❌ Terjadi Error saat menjalankan YOLO: {e}")

# ==================== MODEL TENSORFLOW ====================
elif model_choice == "TensorFlow - ResNet50":
    st.header("🧠 Model TensorFlow - ResNet50")
    
    model = None
    
    # Load TensorFlow model
    @st.cache_resource
    def load_tensorflow():
        tf_path = check_and_load_file(f'model/{TF_MODEL_FILENAME}', TF_MODEL_FILENAME)
        if tf_path:
            try:
                # Muat model
                return tf.keras.models.load_model(tf_path)
            except Exception as e_load:
                st.warning(f"⚠ Gagal load model H5: {str(e_load)[:150]}...")
                st.error("❌ Model tidak bisa dimuat karena Error Internal (Kompatibilitas atau Custom Object).")
                st.info("Coba simpan ulang model H5 Anda di lingkungan pelatihan.")
        return None

    with st.spinner("Loading TensorFlow model..."):
        model = load_tensorflow()
    
    if model is not None:
        st.success("✅ Model berhasil dimuat!")
        
        # Info model
        with st.sidebar.expander("📊 Architecture Model"):
            stream = StringIO()
            model.summary(print_fn=lambda x: stream.write(x + '\n'))
            st.text(stream.getvalue())
        
        # Upload gambar
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
                    
                    # Preprocess KHUSUS UNTUK MENCEGAH VALUEERROR
                    img_array = np.array(image.resize((224, 224))).astype(np.float32) # Pastikan tipe float32
                    
                    # Pastikan 3 channels
                    if img_array.ndim == 2:
                         img_array = np.stack((img_array,)*3, axis=-1)
                    
                    # Convert RGBA to RGB jika perlu
                    if img_array.shape[-1] == 4:
                        img_array = img_array[:, :, :3]
                        
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Preprocessing khusus ResNet50 (zero-centering dengan ImageNet mean)
                    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                    
                    # Prediksi
                    predictions = model.predict(img_array, verbose=0)
                    
                    with col2:
                        st.subheader("🎯 Hasil Prediksi")
                        
                        # Asumsi: model Anda mengklasifikasikan 2 kelas (Hyena/Cheetah)
                        predicted_index = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_index]
                        
                        # --- GANTI INI DENGAN NAMA KELAS ANDA ---
                        CLASSES = ['Cheetah', 'Hyena'] 
                        predicted_class_name = CLASSES[predicted_index]
                        # ----------------------------------------

                        st.metric("Kelas Prediksi", predicted_class_name)
                        st.metric("Confidence", f"{confidence:.2%}")
                        
                        # Tampilkan semua probabilitas
                        with st.expander("📊 Lihat Semua Probabilitas"):
                            for i, prob in enumerate(predictions[0]):
                                st.progress(float(prob), text=f"{CLASSES[i]}: {prob:.4f}")
    
    else: # Model gagal dimuat
        st.info("⚠ Tidak dapat melanjutkan karena model TensorFlow tidak berhasil dimuat.")

# Footer
st.markdown("---")
st.markdown("📌 Dibuat oleh Balqis Isaura** | Powered by Streamlit 🚀")
