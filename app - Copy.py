import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model import TrafficSignCNN
from labels import labels
import os

# Set page config
st.set_page_config(page_title="🚦 Traffic Sign Classifier", page_icon="🚧", layout="centered")

# Title with emoji and style
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🚦 Traffic Sign Recognition App</h1>", unsafe_allow_html=True)
st.markdown("### 🤖 Upload a traffic sign image and let the AI identify it!")

# Device setup
device = torch.device('cpu')

# Load model
num_classes = 43
model = TrafficSignCNN(num_classes=num_classes)
model.load_state_dict(torch.load("outputs/best_model.pth", map_location=device))
model.eval()
model.to(device)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Upload image
uploaded_file = st.file_uploader("📤 Upload a traffic sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

    # Preprocess
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        class_idx = predicted.item()
        class_name = labels[class_idx]
        conf_percent = confidence.item() * 100

    # Result box
    st.markdown("---")
    st.markdown(f"""
        <div style='padding: 1.5rem; background-color: #e8f4fa; border-radius: 10px; text-align: center;'>
            <h2 style='color: #1f77b4;'>🧠 Predicted Class:</h2>
            <h1 style='color: #2ca02c;'>{class_name}</h1>
            <p style='font-size: 18px;'>🔍 Confidence: <strong>{conf_percent:.2f}%</strong></p>
        </div>
    """, unsafe_allow_html=True)

# Sample images
with st.expander("📸 Try with Sample Traffic Signs"):
    cols = st.columns(4)
    sample_path = "sample_signs"
    sample_images = os.listdir(sample_path) if os.path.exists(sample_path) else []

    for idx, filename in enumerate(sample_images):
        with cols[idx % 4]:
            st.image(os.path.join(sample_path, filename), caption=filename, use_column_width=True)

# About Section
st.markdown("---")
st.markdown("""
### ℹ️ About This App

This app uses a **Convolutional Neural Network (CNN)** trained on the [GTSRB dataset](https://benchmark.ini.rub.de/) to identify 43 types of traffic signs.  
It helps build safer autonomous vehicles and supports road safety innovations.

*Built with ❤️ using PyTorch, Streamlit & Computer Vision.*
""")
