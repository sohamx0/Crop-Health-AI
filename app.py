from flask import Flask, render_template, request, jsonify
import ollama
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, static_folder='assets', static_url_path='/assets')

# --- 1. LOAD YOUR LOCAL DISEASE DETECTION MODEL ---
try:
    disease_model = load_model('plant_disease_model.h5')
    DISEASE_CLASS_NAMES = sorted(['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'])
    print("✅ Local disease detection model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading local disease model: {e}")
    disease_model = None


# --- 2. UPDATED OLLAMA FERTILIZER FUNCTION ---
def get_ollama_fertilizer_recommendation(plant_name, disease_name, is_healthy, lat, lon):
    health_status = "Healthy" if is_healthy else f"Diseased with: {disease_name}"

    # --- ✨ THIS PROMPT IS NOW CHANGED TO ASK FOR A BRIEF SUMMARY ✨ ---
    prompt = f"""
        You are an expert agronomist AI. A user's plant has been diagnosed.
        - Plant Species: {plant_name}
        - Health Status: {health_status}
        - User's Location: Pune, India area.

        Your Task: Based on the data, provide a very brief, scannable summary of the top 3-4 most important actions for fertilizer and disease management.
        
        Instructions:
        - Use bullet points.
        - Do not use markdown for bolding.
        - Keep the entire response under 80 words.
    """

    try:
        response = ollama.chat(
            model='llama3.1:8b',
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "Could not connect to local Ollama server. Please make sure Ollama is running on your computer."


# --- API Prediction Route ---
@app.route('/predict_disease', methods=['POST'])
def predict_disease_route():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if not lat or not lon: return jsonify({'error': 'Location data is missing.'}), 400
    if not disease_model: return jsonify({'error': 'Disease model is not loaded.'}), 500
    
    try:
        # Step 1: Diagnosis with your Local Model
        img = Image.open(file.stream).convert('RGB').resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = disease_model.predict(img_array)
        pred_index = np.argmax(prediction[0])
        pred_class_name = DISEASE_CLASS_NAMES[pred_index]
        confidence = np.max(prediction[0]) * 100
        
        display_name = pred_class_name.replace('___', ' - ').replace('_', ' ')
        is_healthy = 'healthy' in pred_class_name.lower()
        plant_name = pred_class_name.split('___')[0]
        disease_name = display_name.split(' - ')[-1]

        # Step 2: Get Fertilizer Recommendation from Ollama
        fertilizer_rec = get_ollama_fertilizer_recommendation(plant_name, disease_name, is_healthy, lat, lon)
        
        prescription = "The plan below is for maintenance and prevention." if is_healthy else "The plan below will help your plant recover."

        return jsonify({
            'disease': display_name, 
            'confidence': f"{confidence:.2f}",
            'is_healthy': is_healthy,
            'prescription': prescription,
            'fertilizer': fertilizer_rec
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'Analysis failed: {e}'}), 500


# --- Other Routes ---
@app.route('/')
def home(): return render_template('index.html')
@app.route('/scanner')
def scanner(): return render_template('vision_checker.html')
@app.route('/library')
def library(): return render_template('library.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)