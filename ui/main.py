import os
import sys
from PIL import Image
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.detection import run_yolo_inference

st.set_page_config(page_title="Child Labour Detector", layout="centered")
st.title("Child Labour Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running YOLOv8 Detection..."):
            result_img, detections, crops = run_yolo_inference(image)
            st.image(result_img, caption="Detected Persons", use_column_width=True)

            if crops:
                st.subheader("Detected Individuals")
                for i, crop in enumerate(crops, 1):
                    st.image(crop, caption=f"Person {i}")
            else:
                st.warning("No people detected in the image.")