import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Skin Condition Detector", layout="centered")
st.title("Skin Condition Detection")
st.write("Upload a skin image and the model will detect and label conditions.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# This model produces low confidence scores overall, so default threshold is
# set much lower than the usual 25% — adjust with the slider as needed.
conf_threshold = st.slider(
    "Confidence threshold", min_value=0.01, max_value=0.50, value=0.10, step=0.01
)

uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running detection..."):
        results = model.predict(image, conf=conf_threshold)
        annotated = results[0].plot()

    annotated_rgb = annotated[:, :, ::-1]
    st.image(annotated_rgb, caption="Detections", use_container_width=True)

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("Detections")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"- **{label}**: {conf:.2%} confidence")
    else:
        st.write("No objects detected at this threshold. Try lowering the slider.")
