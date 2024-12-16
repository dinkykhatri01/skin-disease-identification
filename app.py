import streamlit as st
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Load the CNN model for image-based predictions
cnn_model = tf.keras.models.load_model('cnn_model.h5')

# Load the RandomForest model and TF-IDF vectorizer for text-based predictions
with open('text_symptom_model.pkl', 'rb') as model_file:
    text_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define class names for the CNN model
class_names = ['Eczema', 'Melanoma', 'Psoriasis']

# Streamlit UI
st.title("Skin Disease Prediction")
st.write("You can upload an image or enter symptoms to get a prediction.")

# Option for user to choose image or text input
input_method = st.selectbox("Choose input method", ("Upload Image", "Enter Symptoms"))

# Handle image-based prediction
if input_method == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((128, 128))  # Resize to the input size of the model
        img_array = image.img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Prediction using CNN model
        predictions = cnn_model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100  # Confidence percentage

        # Display results
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write(f"**Predicted Disease:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")

# Handle text-based prediction
elif input_method == "Enter Symptoms":
    user_input = st.text_area("Enter symptoms (e.g., 'Itching, red patches, dry skin')")

    if user_input:
        # Preprocess the input using the saved TF-IDF vectorizer
        input_vector = tfidf_vectorizer.transform([user_input])

        # Prediction using the RandomForest model
        predicted_disease = text_model.predict(input_vector)[0]

        # Display the result
        st.write(f"**Predicted Disease:** {predicted_disease}")
