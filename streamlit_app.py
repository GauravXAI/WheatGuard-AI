import json
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms, models
from torch import nn

IMG_SIZE = 224
MODEL_PATH = "wheat_disease_resnet18.pth"
CLASS_PATH = "class_names.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Wheat Disease Detection",
    page_icon="🌾",
    layout="centered"
)

st.title("🌾 Wheat Disease Detection System")
st.write("Upload a wheat leaf image to detect the disease class.")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, len(class_names))
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload wheat leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100

    st.subheader("Prediction Result")
    st.success(f"Disease: {predicted_class}")
    st.info(f"Confidence: {confidence_score:.2f}%")

    st.subheader("Class Probabilities")

    probs = probabilities.cpu().numpy()[0]

    for class_name, prob in zip(class_names, probs):
        st.write(f"{class_name}: {prob * 100:.2f}%")
        st.progress(float(prob))