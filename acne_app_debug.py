import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Acne Lesion Detector", page_icon="🌿", layout="centered")

st.title("Acne Lesion Detector — DEBUG MODE")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.write(f"**Model task:** {model.task}")
st.write(f"**Model classes:** {model.names}")

uploaded_file = st.file_uploader("Upload a facial skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.write(f"**Image size:** {image.size}, mode: {image.mode}")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Detecting..."):
        results = model.predict(image, conf=0.001, verbose=True)
        boxes = results[0].boxes

    st.write(f"**Raw boxes at conf=0.001:** {len(boxes) if boxes is not None else 0}")
    if boxes is not None and len(boxes) > 0:
        confs = sorted(boxes.conf.tolist(), reverse=True)
        st.write(f"**Top 10 confidences:** {[round(c, 3) for c in confs[:10]]}")
        st.write(f"**Boxes above 0.20:** {sum(1 for c in confs if c > 0.20)}")
    else:
        st.write("Zero boxes returned even at conf=0.001 — this points to a model/environment issue.")
