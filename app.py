import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image # Pillow library for image manipulation

# Initialize Flask application
app = Flask(__name__, static_folder='.', static_url_path='') # Serve static files from current dir

# Load the pre-trained model
try:
    model = load_model('waste-classification-model.h5')
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # Handle the error appropriately, maybe exit or use a dummy model
    model = None # Set model to None if loading fails

# Define the class labels (make sure these match your model's training)
class_labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

# Function to preprocess the image data received from the frontend
def preprocess_image_data(image_data):
    """
    Preprocesses raw image data (bytes) for the model.
    Decodes image, resizes, converts to array, normalizes, and adds batch dimension.
    """
    try:
        # Open image from bytes using Pillow
        image = Image.open(io.BytesIO(image_data))

        # Ensure image is RGB (some cameras might send RGBA or grayscale)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize to the target size your model expects (e.g., 32x32)
        image = image.resize((32, 32))

        # Convert PIL image to NumPy array
        image_array = img_to_array(image, dtype=np.uint8) # Keep as uint8 initially if needed

        # Normalize the image data (scale pixel values to 0-1)
        image_array = np.array(image_array, dtype=np.float32) / 255.0

        # Add batch dimension (model expects input shape like (batch_size, height, width, channels))
        image_batch = image_array[np.newaxis, ...]
        print(f"Preprocessed image shape: {image_batch.shape}") # Debug output
        return image_batch
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Route to serve the main HTML page
@app.route('/')
def index():
    # Sends index.html from the static_folder (current directory)
    return app.send_static_file('index.html')

# Route for prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Missing "image" key in JSON data'}), 400

    # --- Image Data Handling ---
    # The image data comes as a base64 encoded string with a prefix (e.g., "data:image/jpeg;base64,...")
    image_data_url = data['image']
    try:
        # Find the start of the base64 string
        header, encoded_data = image_data_url.split(',', 1)
        # Decode the base64 string into bytes
        image_bytes = base64.b64decode(encoded_data)
    except Exception as e:
        print(f"Error decoding base64 image data: {e}")
        return jsonify({'error': 'Invalid image data format'}), 400

    # --- Preprocessing ---
    processed_image = preprocess_image_data(image_bytes)
    if processed_image is None:
        return jsonify({'error': 'Failed to preprocess image'}), 500

    # --- Prediction ---
    try:
        prediction = model.predict(processed_image)
        print(f"Raw Prediction Output: {prediction}") # Debug output

        # Get the index and label of the highest probability class
        predicted_index = np.argmax(prediction[0])
        if predicted_index < len(class_labels):
            predicted_class = class_labels[predicted_index]
            confidence = float(prediction[0][predicted_index]) # Get confidence score
            print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}") # Debug output
            # Return the result as JSON
            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}"
                })
        else:
             return jsonify({'error': 'Prediction index out of bounds'}), 500

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

# Run the Flask app
if __name__ == '__main__':
    # Run on 0.0.0.0 to make it accessible on your network
    # Use debug=True for development (provides detailed errors and auto-reloads)
    # For production, use a proper WSGI server like Gunicorn or Waitress
    app.run(host='0.0.0.0', port=5000, debug=True)