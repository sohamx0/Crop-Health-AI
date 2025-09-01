# vision_server.py

# Step 1: Make sure you have the necessary libraries
# pip install Flask Flask-Cors
# pip install tensorflow
# pip install numpy
# pip install Pillow

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

# --- What this script does ---
# 1. Creates a web server using Flask.
# 2. Loads our trained plant disease model (`plant_disease_model.h5`).
# 3. Creates an "endpoint" (a URL) that can accept image uploads.
# 4. When an image is uploaded, it preprocesses it and asks the model for a prediction.
# 5. Sends the prediction (disease name and confidence) back to the user.

# --- Step 2: Initialize the Flask App ---
app = Flask(__name__)
CORS(app) # This allows our web page to talk to the server

# --- Step 3: Load the AI Model and Class Names ---
model = None
class_names = None

def load_model():
    """Load the trained model and class names into memory."""
    global model, class_names
    model_path = 'plant_disease_model.h5'
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # Get class names from the training directory structure
    train_dir = 'New Plant Diseases Dataset(Augmented)/train'
    class_names = sorted(os.listdir(train_dir))
    print(f"Found {len(class_names)} classes: {class_names}")

# --- Step 4: Create the Prediction Endpoint ---
@app.route('/predict_disease', methods=['POST'])
def predict():
    """Handles the image upload and returns a prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file:
        try:
            # Read the image file into memory
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes))

            # Ensure image is in RGB format
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Preprocess the image to match model's input requirements
            img = img.resize((128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) # Create a batch
            img_array /= 255.0  # Normalize

            # Make prediction
            prediction = model.predict(img_array)
            
            # Get the top prediction
            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index] * 100)

            print(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}%")

            # Return the result as JSON
            return jsonify({
                'disease': predicted_class_name.replace('___', ' '),
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'Error processing the image'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

# --- Step 5: Run the Server ---
if __name__ == '__main__':
    # Load the model when the server starts
    load_model()
    # Run the app on port 5001 to avoid conflicts with our other server
    app.run(port=5001, debug=True)
