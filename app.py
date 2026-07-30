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

st.write(f"Model task: {model.task}")
st.write(f"Model classes: {model.names}")

uploaded_file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.write(f"Image size: {image.size}, mode: {image.mode}")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Running detection..."):
        # conf=0.001 to catch essentially anything the model produces
        results = model.predict(image, conf=0.001, verbose=True)
        annotated = results[0].plot()

    annotated_rgb = annotated[:, :, ::-1]
    st.image(annotated_rgb, caption="Detections (conf >= 0.001)", use_container_width=True)

    boxes = results[0].boxes
    st.write(f"Number of raw boxes returned: {len(boxes) if boxes is not None else 0}")

    if boxes is not None and len(boxes) > 0:
        confs = boxes.conf.tolist()
        st.write(f"Highest confidence score: {max(confs):.4f}")
        st.subheader("Detections")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            st.write(f"- **{label}**: {conf:.4%} confidence")
    else:
        st.write("Truly zero boxes returned by the model, even at conf=0.001.")
