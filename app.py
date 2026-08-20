import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Acne Lesion Detector", page_icon="🌿", layout="centered")

# ---------- Custom styling ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #FBF3EF;
    color: #2E2622;
    font-family: 'Inter', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { visibility: hidden; }

.sk-header {
    padding: 8px 0 16px 0;
    border-bottom: 1px solid rgba(232, 131, 107, 0.25);
    margin-bottom: 10px;
}
.sk-title {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    font-size: 2.1rem;
    color: #2E2622;
    letter-spacing: -0.01em;
}
.sk-sub {
    color: #8C7A73;
    font-size: 0.97rem;
    margin-top: 6px;
}
.sk-warn {
    font-size: 0.82rem;
    color: #9C5A46;
    background: rgba(232, 131, 107, 0.08);
    border-left: 2px solid #E8836B;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin: 16px 0 26px 0;
    line-height: 1.5;
}
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(232, 131, 107, 0.4);
    border-radius: 12px;
    background: #FFFFFF;
    padding: 8px;
}
.sk-result-card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 20px;
    box-shadow: 0 2px 14px rgba(46, 38, 34, 0.06);
}
.sk-count-label {
    font-family: 'Fraunces', serif;
    font-size: 1.3rem;
    font-weight: 600;
    color: #E8836B;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="sk-header">
    <div class="sk-title">Acne Lesion Detector</div>
    <div class="sk-sub">Upload a close-up facial photo to detect and mark individual acne spots.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sk-warn">
This is a demonstration model, not a medical diagnostic tool. Please consult a dermatologist for real skin concerns.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload a facial skin image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Detecting..."):
        results = model.predict(image, conf=0.20)
        boxes = results[0].boxes

    # ---------- Draw circle markers with confidence labels ----------
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                   size=max(14, image.width // 55))
    except Exception:
        font = ImageFont.load_default()

    count = 0
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            r = max(x2 - x1, y2 - y1) / 2

            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(24, 130, 90, 255), width=3)

            label = f"{conf:.2f}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            lx, ly = cx - text_w / 2, cy - r - text_h - 6
            draw.rectangle([lx - 4, ly - 2, lx + text_w + 4, ly + text_h + 4], fill=(46, 38, 34, 160))
            draw.text((lx, ly - text_bbox[1]), label, fill=(255, 255, 255, 255), font=font)
            count += 1

    st.image(annotated, caption="Detected Lesions", use_container_width=True)

    st.markdown('<div class="sk-result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sk-count-label">{count} spot{"s" if count != 1 else ""} detected</div>', unsafe_allow_html=True)
    if count == 0:
        st.write("No acne spots detected at this confidence level.")
    st.markdown('</div>', unsafe_allow_html=True)
