import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from src.model import get_model

# =========================
# DEVICE
# =========================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# =========================
# CLASS LABELS
# =========================
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =========================
# LOAD MODEL
# =========================
model = get_model(len(classes), device)
model.load_state_dict(torch.load("/Users/lomash/programmingPy/practice_2026/praktika_emoji/models/best_model.pth", map_location=device))
model.to(device)
model.eval()

# =========================
# TRANSFORM (ВАЖНО!)
# =========================
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# =========================
# UI
# =========================
st.title("😄 Emotion Recognition App")

st.write("Загрузи изображение лица, и модель определит эмоцию")

file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file).convert('RGB')  # Гарантирует, что картинка всегда в RGB
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # preprocessing
    x = transform(image).unsqueeze(0).to(device)

    # prediction
    with torch.no_grad():
        output = model(x)
        pred = output.argmax(dim=1).item()

    st.subheader(f"Prediction: {classes[pred]}")