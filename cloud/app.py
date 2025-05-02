import io
import base64
import numpy as np
import os
import time
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
# Import VGG16 preprocessing specifically
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from PIL import Image
from flask_cors import CORS
import traceback # For better error logging

# --- Configuration ---
MODEL_FILENAME = 'model.hdf5' # Your trained model file
IMAGE_WIDTH = 224             # Match training image size
IMAGE_HEIGHT = 224            # Match training image size
UPLOAD_FOLDER = 'user_corrections' # Folder for online update images

CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

app = Flask(__name__)
CORS(app) 

if not os.path.exists(UPLOAD_FOLDER):
    try:
        os.makedirs(UPLOAD_FOLDER)
        print(f"Created directory: {UPLOAD_FOLDER}")
    except OSError as e:
        print(f"Error creating directory {UPLOAD_FOLDER}: {e}")

# --- Load Model ---
loaded_model = None

def load_app_model():
    """Loads the model into the global variable."""
    global loaded_model
    if loaded_model is None: # Load only once
        try:
            print(f"Loading model '{MODEL_FILENAME}'...")
            loaded_model = load_model(MODEL_FILENAME, compile=False)
            print(f"Successfully loaded model '{MODEL_FILENAME}'.")
            dummy_input = np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
            _ = loaded_model.predict(dummy_input)
            print("Model warm-up prediction complete.")
        except Exception as e:
            print(f"FATAL: Error loading model '{MODEL_FILENAME}': {e}")
            traceback.print_exc()
            loaded_model = None
    return loaded_model

# --- Image Preprocessing Function ---
# Updated for 224x224 and VGG16 preprocessing
def preprocess_image_data(image_data):
    """Preprocesses raw image bytes for VGG16."""
    try:
        image = Image.open(io.BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize to the model's expected input size
        image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        image_array = img_to_array(image) # Converts to float32 numpy array
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        # Apply VGG16 specific preprocessing
        processed_batch = vgg16_preprocess_input(image_batch)
        # print(f"Preprocessed image shape: {processed_batch.shape}")
        return processed_batch
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        traceback.print_exc()
        return None

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    print("Received request at /predict")
    model = load_app_model() # Ensure model is loaded
    if model is None:
         print("Prediction failed: Model not available.")
         return jsonify({'error': 'Model failed to load on server'}), 500

    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Missing "image" key'}), 400

    # --- Image Handling & Preprocessing ---
    image_data_url = data['image']
    try:
        header, encoded_data = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded_data)
        processed_image = preprocess_image_data(image_bytes)
        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500
    except Exception as e:
        print(f"Error decoding or preprocessing image: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Invalid image data or preprocessing failed'}), 400

    # --- Prediction ---
    try:
        prediction_result = model.predict(processed_image)
        predicted_index = np.argmax(prediction_result[0])

        if 0 <= predicted_index < len(CLASS_LABELS):
            predicted_class = CLASS_LABELS[predicted_index]
            confidence = float(prediction_result[0][predicted_index])
            print(f"Prediction: {predicted_class} ({confidence:.2f})")
            # Return prediction and confidence
            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}"
            })
        else:
             print(f"Error: Prediction index {predicted_index} out of bounds (Num classes: {len(CLASS_LABELS)})")
             return jsonify({'error': 'Prediction index out of bounds'}), 500

    except Exception as e:
        print(f"Error during prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed on server: {e}'}), 500

# Endpoint for receiving corrections (for online updating)
@app.route('/submit_correction', methods=['POST'])
def submit_correction():
    print("Received request at /submit_correction")
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    if 'image' not in data or 'label' not in data:
        return jsonify({'error': 'Missing "image" or "label" key'}), 400

    image_data_url = data['image']
    # Ensure label consistency (e.g., lowercase)
    correct_label = data['label'].lower()

    # Validate label against known classes
    if correct_label not in CLASS_LABELS:
         print(f"Invalid label received: {correct_label}")
         return jsonify({'error': f'Invalid label provided: {correct_label}'}), 400

    try:
        # Decode image
        header, encoded_data = image_data_url.split(',', 1)
        image_bytes = base64.b64decode(encoded_data)

        # Create a unique filename: Label_Timestamp.jpg
        timestamp = int(time.time() * 1000)
        filename = f"{correct_label}_{timestamp}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        # Save using Pillow to ensure format consistency (recommended)
        img_pil = Image.open(io.BytesIO(image_bytes))
        if img_pil.mode != 'RGB':
            img_pil = img_pil.convert('RGB')
        img_pil.save(filepath, "JPEG")
        print(f"Saved correction: {filepath}")

        return jsonify({'message': 'Correction received successfully'}), 200

    except Exception as e:
        print(f"Error saving correction: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Failed to decode or save correction'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    load_app_model()
    print("Starting Flask server via app.run() (for testing only)...")
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    load_app_model()
    print("Flask app initialized for Gunicorn.")