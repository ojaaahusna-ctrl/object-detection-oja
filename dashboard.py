import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# --- Konfigurasi Awal ---
st.set_page_config(page_title="Image Classifier Dashboard", layout="wide")

# =======================================================
# Fungsi Pemuatan Model (Menggunakan Cache Streamlit)
# =======================================================

# Model 1: Hyena vs Cheetah (H5/TensorFlow)
@st.cache_resource
def load_hyena_cheetah_model():
    model = tf.keras.models.load_model('model/Raudhatul Husna_laporan2.h5')
    return model

# Model 2: Hot Dog vs Not Hot Dog (PT/PyTorch)
@st.cache_resource
def load_hotdog_model():
    # Model PyTorch umumnya dimuat menggunakan torch.load
    model = torch.load('model/best.pt', map_location=torch.device('cpu'))
    model.eval() # Atur ke mode evaluasi/inferensi
    return model

# Muat semua model
model_h5 = load_hyena_cheetah_model()
model_pt = load_hotdog_model()

# Daftar Kelas
HYENA_CHEETAH_CLASSES = ['Cheetah', 'Hyena'] # SESUAIKAN URUTANNYA DENGAN MODEL ANDA
HOTDOG_CLASSES = ['Not Hot Dog', 'Hot Dog'] # SESUAIKAN URUTANNYA DENGAN MODEL ANDA



# =======================================================
# Fungsi Prediksi Model H5 (Hyena/Cheetah)
# =======================================================
def predict_h5(image, model):
    # Preprocessing untuk Keras/TensorFlow (contoh ukuran 224x224)
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambah dimensi batch
    
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]
    
    result_class = HYENA_CHEETAH_CLASSES[predicted_index]
    return result_class, confidence

# =======================================================
# Fungsi Prediksi Model PT (Hot Dog/Not Hot Dog)
# =======================================================
def predict_pt(image, model):
    # Preprocessing untuk PyTorch (contoh ImageNet standard)
    pt_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = pt_transform(image).unsqueeze(0) # Tambah dimensi batch
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        # Ambil hasil dengan probabilitas tertinggi
        confidence, predicted_index_tensor = torch.max(probabilities, 1)
        predicted_index = predicted_index_tensor.item()
        confidence = confidence.item()
        
    result_class = HOTDOG_CLASSES[predicted_index]
    return result_class, confidence


# =======================================================
# Tampilan Streamlit
# =======================================================

st.title("Dual Model Image Classification Dashboard 🖼️")
st.markdown("Pilih model dan unggah gambar untuk klasifikasi.")

# Pilihan Model menggunakan radio button
model_choice = st.radio(
    "Pilih Model Klasifikasi:",
    ('Hyena/Cheetah (H5/TensorFlow)', 'Hot Dog/Not Hot Dog (PT/PyTorch)'),
    horizontal=True
)

# Upload Gambar
uploaded_file = st.file_uploader("Unggah Gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Tampilkan Gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar yang Diunggah.', use_column_width=True)
    st.write("")
    
    # 2. Lakukan Prediksi Berdasarkan Pilihan
    st.subheader("Hasil Prediksi")
    
    with st.spinner('Memproses prediksi...'):
        if model_choice == 'Hyena/Cheetah (H5/TensorFlow)':
            result_class, confidence = predict_h5(image, model_h5)
            st.success(f"**Klasifikasi Model H5:** {result_class}")
            st.info(f"Keyakinan (Confidence): **{confidence * 100:.2f}%**")
            
        elif model_choice == 'Hot Dog/Not Hot Dog (PT/PyTorch)':
            result_class, confidence = predict_pt(image, model_pt)
            st.success(f"**Klasifikasi Model PT:** {result_class}")
            st.info(f"Keyakinan (Confidence): **{confidence * 100:.2f}%**")

# Jika menggunakan GitHub, pastikan Streamlit dijalankan dengan:
# streamlit run dashboard.py
