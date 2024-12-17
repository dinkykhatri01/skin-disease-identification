import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model
import os
import speech_recognition as sr

# --------------------- Load Models and Data ---------------------
# Load CNN model for image-based disease classification
cnn_model = load_model('cnn_model.h5')

# Load segmentation model for preprocessing (optional, but not shown)
segmentation_model = load_model('segmentation_model.h5')

# Load symptom-based text classification model
with open('text_symptom_model.pkl', 'rb') as file:
    text_model = pickle.load(file)

# Load TF-IDF Vectorizer for text input
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

# --------------------- Streamlit Page Config ---------------------
st.set_page_config(
    page_title="Skin Disease Detector",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Add custom styles
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #4A90E2;
        text-align: center;
    }
    .subtitle {
        font-size: 24px;
        font-weight: 600;
        color: #444444;
        text-align: center;
    }
    .footer {
        font-size: 14px;
        color: #777777;
        text-align: center;
    }
    .uploaded-image {
        border: 3px solid #4A90E2;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------- Page Layout ---------------------
st.markdown('<p class="title">Skin Disease Detector 🩺</p>', unsafe_allow_html=True)
st.write("Upload an image for skin disease prediction or enter your symptoms for analysis.")

# --------------------- Image-Based Disease Prediction ---------------------
st.markdown('<p class="subtitle">Image-Based Prediction</p>', unsafe_allow_html=True)
uploaded_image = st.file_uploader("Upload an image of the affected skin (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    img = image.resize((128, 128))  # Resize image to model input
    img = np.array(img) / 255.0     # Normalize pixel values
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True, output_format="auto")
    st.markdown('<hr>', unsafe_allow_html=True)
    
    # Predict using CNN model
    st.write("Predicting disease...")
    processed_img = preprocess_image(image)
    predictions = cnn_model.predict(processed_img)

    # Display result
    class_names = ["Melanoma", "Psoriasis", "Eczema", "Healthy"]
    predicted_class = class_names[np.argmax(predictions)]
    infection_percentage = f"{np.max(predictions) * 100:.2f}%"

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Infection Percentage: {infection_percentage}")

# --------------------- Text-Based Symptom Prediction ---------------------
st.markdown('<p class="subtitle">Symptom-Based Prediction</p>', unsafe_allow_html=True)

# Initialize session state for symptoms
if "symptoms" not in st.session_state:
    st.session_state["symptoms"] = ""

# Text area for symptom input
st.session_state["symptoms"] = st.text_area(
    "Enter your symptoms (e.g., red patches, itchy skin, blisters):",
    value=st.session_state["symptoms"]
)

# Voice input button
st.write("OR")
if st.button("Speak Symptoms"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak your symptoms.")
        try:
            audio = r.listen(source, timeout=5)
            spoken_text = r.recognize_google(audio)
            st.success(f"Recognized Symptoms: {spoken_text}")
            st.session_state["symptoms"] = spoken_text  # Update session state with spoken symptoms
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech. Please try again.")
        except sr.RequestError as e:
            st.error(f"Speech Recognition service is unavailable. {e}")

# Predict button for text-based symptoms
if st.button("Predict Disease"):
    if st.session_state["symptoms"]:
        # Process symptom input
        text_features = tfidf_vectorizer.transform([st.session_state["symptoms"]])
        text_prediction = text_model.predict(text_features)[0]

        st.success(f"Predicted Disease: {text_prediction}")
    else:
        st.warning("Please enter symptoms or use the voice input to predict the disease.")
