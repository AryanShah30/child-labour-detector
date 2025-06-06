import streamlit as st
from PIL import Image
import numpy as np
import sys
import os

from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.detection import run_yolo_inference

# Load trained model
age_model = load_model(os.path.join("models", "age_group_classifier.h5"))

def preprocess_crop(crop):
    # If YOLO gave us a NumPy array, turn it into a PIL Image
    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop)
    # Now we can safely resize
    crop = crop.resize((128, 128))
    crop_array = img_to_array(crop) / 255.0
    return np.expand_dims(crop_array, axis=0)

def classify_age(crop):
    processed = preprocess_crop(crop)
    prediction = age_model.predict(processed)[0][0]
    st.write("Raw model prediction:", prediction)
    return "Child" if prediction < 0.5 else "Adult"

st.set_page_config(page_title="Child Labour Detector", layout="centered")
st.title("Child Labour Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Detection"):
        with st.spinner("Running YOLOv8"):
            result_img, detections, crops = run_yolo_inference(image)
            st.image(result_img, caption="Detected Persons", use_column_width=True)

            if crops:
                st.subheader("Cropped Persons with Age Group")
                for i, crop in enumerate(crops):
                    age_group = classify_age(crop)
                    st.image(crop, caption=f"Person {i+1}: {age_group}")
            else:
                st.warning("No people detected.")
                st.stop()
