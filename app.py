# app.py

# Step 1: Install Flask
# Open your terminal and run: pip install Flask

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS # Import CORS

# --- What this script does ---
# 1. Loads our pre-trained AI model (`crop_model.joblib`).
# 2. Creates a simple web server using Flask.
# 3. Defines an API endpoint `/predict` that listens for requests from our webpage.
# 4. When it gets data, it uses the model to make a prediction and sends the result back.

# --- Create the Flask App ---
app = Flask(__name__)
CORS(app) # This is important to allow our HTML page to talk to the server

# --- Load the Trained Model ---
# Make sure 'crop_model.joblib' is in the same folder as this script.
try:
    model = joblib.load('crop_model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("--- ERROR ---")
    print("The 'crop_model.joblib' file was not found. Please run main.py to create it.")
    model = None

# --- Define the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'}), 500

    # 1. Get the data from the POST request sent by the webpage
    data = request.get_json()
    print(f"Received data: {data}")

    # 2. Prepare the data for the model
    # The model expects the data in a specific order:
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Extract values in the correct order, providing a default of 0 if a key is missing
    input_data = [
        float(data.get('nitrogen', 0)),
        float(data.get('phosphorus', 0)),
        float(data.get('potassium', 0)),
        float(data.get('temperature', 0)),
        float(data.get('humidity', 0)),
        float(data.get('ph', 0)),
        # Assuming rainfall is not in the UI yet, so we'll use a typical average
        float(data.get('rainfall', 100)) 
    ]
    
    # Create a DataFrame for the model
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # 3. Make a prediction
    prediction = model.predict(input_df)
    
    # The output is an array, so we get the first element
    crop_recommendation = prediction[0]
    print(f"Prediction: {crop_recommendation}")

    # 4. Send the prediction back to the webpage as JSON
    return jsonify({'recommendation': crop_recommendation})

# --- Run the App ---
if __name__ == '__main__':
    # The server will run on http://127.0.0.1:5000
    app.run(debug=True)

