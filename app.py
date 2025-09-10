import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

try:
    model = joblib.load('crop_model.joblib')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("--- ERROR ---")
    print("The 'crop_model.joblib' file was not found. Please run main.py to create it.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please check the server logs.'}), 500

    data = request.get_json()
    print(f"Received data: {data}")

    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    
    input_data = [
        float(data.get('nitrogen', 0)),
        float(data.get('phosphorus', 0)),
        float(data.get('potassium', 0)),
        float(data.get('temperature', 0)),
        float(data.get('humidity', 0)),
        float(data.get('ph', 0)),
        float(data.get('rainfall', 100)) 
    ]
    
    input_df = pd.DataFrame([input_data], columns=feature_names)

    prediction = model.predict(input_df)
    
    crop_recommendation = prediction[0]
    print(f"Prediction: {crop_recommendation}")

    return jsonify({'recommendation': crop_recommendation})

if __name__ == '__main__':
    # The server will run on http://127.0.0.1:5000
    app.run(debug=True)

