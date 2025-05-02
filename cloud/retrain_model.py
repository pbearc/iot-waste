import os
import shutil 
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# Import VGG16 preprocessing, consistent with app.py
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
import glob 

MODEL_PATH = 'model.hdf5' 
BACKUP_MODEL_PATH = 'model-backup.hdf5' 
TEMP_MODEL_PATH = 'model-updated.hdf5' 
CORRECTIONS_FOLDER = 'user_corrections'
PROCESSED_FOLDER = 'processed_corrections' 
IMG_WIDTH, IMG_HEIGHT = 224, 224 
BATCH_SIZE = 8 
EPOCHS = 5 
MIN_SAMPLES_FOR_RETRAIN = 1 
CLASS_LABELS = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def preprocess_image_for_retrain(image_path):
    """Preprocesses an image file for fine-tuning using VGG16 preprocessing."""
    try:
        img = load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
        img_array = img_to_array(img) # Shape (height, width, channels)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, height, width, channels)
        processed_img = vgg16_preprocess_input(img_array) # Apply VGG16 preprocessing
        # print(f"Processed {image_path} shape: {processed_img.shape}") # Debug
        return processed_img[0] # Return just the single image array (remove batch dim)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def load_corrections(folder_path):
    """Loads correction images and labels from the specified folder."""
    images = []
    labels = []
    class_indices = {name: i for i, name in enumerate(CLASS_LABELS)}

    print(f"Searching for corrections in {folder_path}...")
    files_to_process = glob.glob(os.path.join(folder_path, '*.jpg')) # Get all JPGs

    if not files_to_process:
        print("No image files found in corrections folder.")
        return None, None, [] # Return empty list of files

    print(f"Found {len(files_to_process)} files.")
    loaded_files = []

    for filepath in files_to_process:
        filename = os.path.basename(filepath)
        try:
            label_name = filename.split('_')[0]
            if label_name in class_indices:
                label_index = class_indices[label_name]
                processed_image = preprocess_image_for_retrain(filepath)

                if processed_image is not None:
                    images.append(processed_image)
                    labels.append(label_index)
                    loaded_files.append(filepath) # Keep track of successfully loaded files
                else:
                     print(f"Skipping {filename} due to processing error.")
            else:
                print(f"Warning: Unknown label '{label_name}' derived from filename {filename}. Skipping.")

        except Exception as e:
            print(f"Error processing filename {filename}: {e}")

    if not images:
        print("No valid correction samples loaded.")
        return None, None, []

    print(f"Loaded {len(images)} valid correction samples.")
    return np.array(images), np.array(labels), loaded_files

def move_processed_files(file_list, destination_folder):
    """Moves files after they have been processed."""
    print(f"Moving {len(file_list)} processed files to {destination_folder}...")
    moved_count = 0
    for filepath in file_list:
        try:
            filename = os.path.basename(filepath)
            destination_path = os.path.join(destination_folder, filename)
            os.rename(filepath, destination_path) 
            moved_count += 1
        except Exception as e:
            print(f"Error moving file {filepath}: {e}")
    print(f"Successfully moved {moved_count} files.")


if __name__ == "__main__":
    print("Starting retraining process...")

    # 1. Load corrections and get list of files processed
    X_new, y_new, processed_file_list = load_corrections(CORRECTIONS_FOLDER)

    if X_new is None or y_new is None or len(X_new) < MIN_SAMPLES_FOR_RETRAIN:
        print(f"Not enough valid correction data found ({len(X_new) if X_new is not None else 0} samples < {MIN_SAMPLES_FOR_RETRAIN} required). Exiting.")
        if processed_file_list:
             move_processed_files(processed_file_list, PROCESSED_FOLDER)
        exit()

    y_new = y_new.astype(np.int32)

    # 2. Check if model path is a directory
    if os.path.isdir(MODEL_PATH):
        print(f"WARNING: {MODEL_PATH} is a directory, not a file. Using model-updated.hdf5 instead.")
        MODEL_PATH = 'model-updated.hdf5'  # Use different path
    
    # 3. Load current model (ensure it compiles for training)
    print(f"Loading current model from {MODEL_PATH}...")
    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully (hopefully with original compile state).")

    except Exception as e:
        print(f"Error loading model compiled: {e}. Trying compile=False and re-compiling...")
        try:
             model = load_model(MODEL_PATH, compile=False)
             model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
             print("Model loaded with compile=False and recompiled.")
        except Exception as e2:
             print(f"FATAL: Could not load or compile model: {e2}")
             # Move processed files before exiting on fatal error
             if processed_file_list:
                 move_processed_files(processed_file_list, PROCESSED_FOLDER)
             exit()

    # 4. Fine-tune the model
    print(f"Fine-tuning model with {len(X_new)} samples for {EPOCHS} epochs...")
    # Shuffle data for better training
    indices = np.arange(X_new.shape[0])
    np.random.shuffle(indices)
    X_new_shuffled = X_new[indices]
    y_new_shuffled = y_new[indices]

    history = model.fit(X_new_shuffled, y_new_shuffled, epochs=EPOCHS, batch_size=BATCH_SIZE)
    print("Fine-tuning complete.")

    # 5. Save the updated model
    print(f"Saving updated model to {TEMP_MODEL_PATH}...")
    try:
        if os.path.isfile(MODEL_PATH):
            print(f"Creating backup of existing model to {BACKUP_MODEL_PATH}")
            shutil.copy2(MODEL_PATH, BACKUP_MODEL_PATH)
        
        # Save the updated model to the temporary path
        model.save(TEMP_MODEL_PATH)
        print(f"Model saved to {TEMP_MODEL_PATH} successfully.")
        
        # 6. Move processed correction files
        move_processed_files(processed_file_list, PROCESSED_FOLDER)

    except Exception as e:
        print(f"Error saving model or moving files: {e}")
        # Clean up temp file if it exists and is a file
        if os.path.isfile(TEMP_MODEL_PATH):
            os.remove(TEMP_MODEL_PATH)

    print("Retraining script finished.")