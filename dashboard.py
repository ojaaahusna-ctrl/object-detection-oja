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
    model = tf.keras.models.load_model('model/hyena_cheetah_model.h5')
    return model

# Model 2: Hot Dog vs Not Hot Dog (PT/PyTorch)
@st.cache_resource
def load_hotdog_model():
    # Model PyTorch umumnya dimuat menggunakan torch.load
    model = torch.load('model/hotdog_nothotdog_model.pt', map_location=torch.device('cpu'))
    model.eval() # Atur ke mode evaluasi/inferensi
    return model

# Muat semua model
model_h5 = load_hyena_cheetah_model()
model_pt = load_hotdog_model()

# Daftar Kelas
HYENA_CHEETAH_CLASSES = ['Cheetah', 'Hyena'] # SESUAIKAN URUTANNYA DENGAN MODEL ANDA
HOTDOG_CLASSES = ['Not Hot Dog', 'Hot Dog'] # SESUAIKAN URUTANNYA DENGAN MODEL ANDA
