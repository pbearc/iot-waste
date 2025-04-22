import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('waste-classification-model.h5')

# Define the class labels
class_labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(32, 32))
    image = img_to_array(image, dtype=np.uint8)
    image = np.array(image) / 255.0
    return image[np.newaxis, ...]

# Streamlit UI
st.title("Waste Classification")
st.write("Upload an image to classify it into one of the waste categories.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = preprocess_image(uploaded_file)

    # Make prediction
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction[0], axis=-1)]

    # Display the image and prediction
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write(f"Predicted Category: {predicted_class}")