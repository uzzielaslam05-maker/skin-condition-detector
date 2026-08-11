import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Facial Skin Disease Classifier", layout="centered")
st.title("Facial Skin Disease Classifier")
st.write("Upload a close-up facial skin photo and the model will predict which condition it most resembles: Acne, Perioral Dermatitis, Milia, or Rosacea.")
st.caption("⚠️ This is a demonstration model, not a medical diagnostic tool. Do not use it to make real medical decisions — consult a dermatologist.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Choose a facial skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Classifying..."):
        results = model.predict(image, verbose=False)
        probs = results[0].probs

    top_class = model.names[probs.top1]
    top_conf = float(probs.top1conf)

    st.subheader(f"Prediction: **{top_class.replace('_', ' ').title()}**")
    st.write(f"Confidence: {top_conf:.2%}")

    st.write("---")
    st.write("All class probabilities:")
    all_probs = {model.names[i]: float(p) for i, p in enumerate(probs.data)}
    for cls, p in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        st.write(f"- {cls.replace('_', ' ').title()}: {p:.2%}")
