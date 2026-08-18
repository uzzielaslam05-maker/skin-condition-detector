import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Facial Skin Classifier", page_icon="🌿", layout="centered")

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
.sk-predicted-label {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #E8836B;
    margin-bottom: 2px;
}
.sk-predicted-conf {
    color: #8C7A73;
    font-size: 0.9rem;
    margin-bottom: 18px;
}
.sk-bar-row {
    margin-bottom: 10px;
}
.sk-bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: #2E2622;
    margin-bottom: 4px;
}
.sk-bar-track {
    background: #F3E9E4;
    border-radius: 6px;
    height: 8px;
    overflow: hidden;
}
.sk-bar-fill {
    background: linear-gradient(90deg, #E8836B, #8FA998);
    height: 100%;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("""
<div class="sk-header">
    <div class="sk-title">Facial Skin Classifier</div>
    <div class="sk-sub">Upload a close-up facial photo to identify the likely skin condition.</div>
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
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing..."):
        results = model.predict(image, verbose=False)
        probs = results[0].probs

    top_class = model.names[probs.top1]
    top_conf = float(probs.top1conf)
    all_probs = {model.names[i]: float(p) for i, p in enumerate(probs.data)}

    # ---------- Stamp the predicted label onto the image itself ----------
    labeled_image = image.copy()
    draw = ImageDraw.Draw(labeled_image, "RGBA")
    label_text = f"{top_class.replace('_', ' ').title()} — {top_conf:.1%}"

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=max(18, image.width // 25))
    except Exception:
        font = ImageFont.load_default()

    text_bbox = draw.textbbox((0, 0), label_text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 10
    box_x0, box_y0 = 12, 12
    box_x1, box_y1 = box_x0 + text_w + pad * 2, box_y0 + text_h + pad * 2

    draw.rectangle([box_x0, box_y0, box_x1, box_y1], fill=(232, 131, 107, 220))
    draw.text((box_x0 + pad, box_y0 + pad - text_bbox[1]), label_text, fill=(255, 255, 255, 255), font=font)

    st.image(labeled_image, caption="Labeled Result", use_container_width=True)

    st.markdown('<div class="sk-result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="sk-predicted-label">{top_class.replace("_", " ").title()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sk-predicted-conf">Predicted with {top_conf:.1%} confidence</div>', unsafe_allow_html=True)

    for cls, p in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        pct = p * 100
        st.markdown(f"""
        <div class="sk-bar-row">
            <div class="sk-bar-label"><span>{cls.replace('_', ' ').title()}</span><span>{p:.1%}</span></div>
            <div class="sk-bar-track"><div class="sk-bar-fill" style="width:{pct}%;"></div></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
