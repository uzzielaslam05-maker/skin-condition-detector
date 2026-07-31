# Skin Condition Detector — Acne Detection

A Streamlit web app that uses a YOLOv8 model to detect acne in uploaded skin images.

⚠️ **Disclaimer:** This model was trained on a small dataset (~90 images total) and may not be highly accurate. This is a demonstration project, not a medical diagnostic tool. Do not use it to make real skincare or medical decisions — consult a dermatologist for actual concerns.

## How it works

1. Upload a close-up, well-lit photo of skin (JPG or PNG).
2. The model (`best.pt`, a YOLOv8 detection model trained to detect the single class `acne`) runs inference on the image.
3. Detected regions are drawn as bounding boxes on the image, along with the confidence score for each detection.

## Tech stack

- **Streamlit** — web app framework
- **Ultralytics YOLOv8** — object detection model
- **OpenCV (headless)** — image processing
- **Pillow** — image loading

## Project structure

```
├── app.py              # Streamlit app
├── best.pt              # YOLOv8 model weights
├── requirements.txt      # Python dependencies
└── packages.txt          # System-level dependencies (for Streamlit Cloud)
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live demo

Deployed on Streamlit Community Cloud — https://skin-condition-detector-bkrfgtnq5pznod5dimzg8o.streamlit.app/


## Known limitations

- Trained on a small dataset (~90 images), so detection reliability is limited.
- Some training images were AI-generated rather than real photographs, which may affect how well the model generalizes to real-world photos.
- Confidence scores are generally lower than typical well-trained models; a threshold of 0.10 is used by default.
