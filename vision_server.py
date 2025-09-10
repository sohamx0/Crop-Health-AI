from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

model = None
class_names = None

def load_model():
    """Load the trained model and class names into memory."""
    global model, class_names
    model_path = 'plant_disease_model.h5'
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    train_dir = 'New Plant Diseases Dataset(Augmented)/train'
    class_names = sorted(os.listdir(train_dir))
    print(f"Found {len(class_names)} classes: {class_names}")

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
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes))

            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize((128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0 

            prediction = model.predict(img_array)

            predicted_class_index = np.argmax(prediction[0])
            predicted_class_name = class_names[predicted_class_index]
            confidence = float(prediction[0][predicted_class_index] * 100)

            print(f"Prediction: {predicted_class_name}, Confidence: {confidence:.2f}%")

            return jsonify({
                'disease': predicted_class_name.replace('___', ' '),
                'confidence': f"{confidence:.2f}"
            })

        except Exception as e:
            print(f"An error occurred: {e}")
            return jsonify({'error': 'Error processing the image'}), 500

    return jsonify({'error': 'An unknown error occurred'}), 500

if __name__ == '__main__':
    load_model()
    app.run(port=5001, debug=True)
