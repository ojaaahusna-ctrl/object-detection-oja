import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import os

# ==========================
# 1️⃣ Load YOLO model (local .pt)
# ==========================
@st.cache_resource
def load_yolo_model():
    model_path = "model/best.pt"
    if not os.path.exists(model_path):
        st.error("❌ File model YOLO tidak ditemukan di folder 'model/'. Pastikan best.pt sudah ada.")
        return None
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    return model


# ==========================
# 2️⃣ Load ResNet50 model (pretrained)
# ==========================
@st.cache_resource
def load_resnet50():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model


# ==========================
# 3️⃣ Fungsi prediksi YOLO
# ==========================
def predict_yolo(model, image):
    results = model(image)
    return results.pandas().xyxy[0]


# ==========================
# 4️⃣ Fungsi prediksi ResNet50
# ==========================
def predict_resnet(model, image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
    probs = torch.nn.functional.softmax(outputs[0], dim=0)
    class_idx = torch.argmax(probs).item()

    # Load labels
    labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    labels = requests.get(labels_url).text.splitlines()
    label = labels[class_idx]
    confidence = probs[class_idx].item()
    return label, confidence


# ==========================
# 5️⃣ Dashboard Streamlit
# ==========================
st.title("📸 Object Detection & Classification Dashboard")
st.write("Gunakan YOLO (best.pt) dan ResNet50 bersama-sama.")

uploaded_image = st.file_uploader("Unggah gambar untuk prediksi", type=["jpg", "png", "jpeg"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Gambar diunggah", use_container_width=True)

    st.write("### 🔍 Hasil Prediksi")

    # Load kedua model
    yolo_model = load_yolo_model()
    resnet_model = load_resnet50()

    if yolo_model:
        st.subheader("Deteksi Objek (YOLO)")
        yolo_results = predict_yolo(yolo_model, image)
        st.dataframe(yolo_results)

    st.subheader("Klasifikasi (ResNet50)")
    label, confidence = predict_resnet(resnet_model, image)
    st.success(f"Prediksi: **{label}** dengan keyakinan {confidence:.2f}")


# ==========================
# 6️⃣ Cara Jalankan (Terminal)
# ==========================
# streamlit run dashboard.py
