from flask import Flask, render_template, request, jsonify
import requests
import json
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
import os
import base64
import io

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

app = Flask(__name__, static_folder='assets', static_url_path='/assets')


# Load API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if not GEMINI_API_KEY:
    print("⚠️  Warning: GEMINI_API_KEY not found in environment variables. Please set it in .env file")
    print("   The AI features (chat and plant validation) will not work without the API key.")
else:
    print(f"✅ Gemini API key loaded (length: {len(GEMINI_API_KEY)})")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else ""

try:
    disease_model = load_model('plant_disease_model.h5')
    DISEASE_CLASS_NAMES = sorted(['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'])
    print("✅ Local disease detection model (ResNet50) loaded successfully!")
except Exception as e:
    print(f"❌ Error loading local disease model: {e}")
    disease_model = None


def ask_gemini(prompt_text):
    if not GEMINI_API_KEY or not GEMINI_URL:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in .env file"
    
    payload = { "contents": [{ "parts": [{"text": prompt_text}] }] }
    try:
        response = requests.post(GEMINI_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            error_msg = f"API Error {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail}"
            except:
                error_msg += f": {response.text[:200]}"
            print(f"Gemini API error: {error_msg}")
            return f"Error: Could not reach AI server. {error_msg}"
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please try again."
    except Exception as e:
        print(f"Exception in ask_gemini: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {e}"

def check_if_plant(image):
    """Check if the image contains a valid plant using Gemini Vision API"""
    if not GEMINI_API_KEY or not GEMINI_URL:
        print("⚠️  Warning: GEMINI_API_KEY not set, skipping plant validation")
        return True  # Allow through if API key not configured
    
    try:
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare payload for Gemini Vision API
        payload = {
            "contents": [{
                "parts": [
                    {
                        "text": "Look at this image and determine if it contains a valid plant (crop, leaf, or plant part). Respond with ONLY 'yes' if it's a plant, or 'no' if it's not a plant. Be strict - only respond 'yes' for actual plants, leaves, or plant parts."
                    },
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": img_base64
                        }
                    }
                ]
            }]
        }
        
        response = requests.post(GEMINI_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
        
        if response.status_code == 200:
            result = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().lower()
            # Check if response indicates it's a plant
            is_valid_plant = 'yes' in result or 'plant' in result or 'leaf' in result or 'crop' in result
            print(f"Gemini plant check result: {result} -> {is_valid_plant}")
            return is_valid_plant
        else:
            error_detail = response.text
            try:
                error_json = response.json()
                error_detail = str(error_json)
            except:
                pass
            print(f"Gemini API error: {response.status_code} - {error_detail}")
            return True  # Default to True if API fails, to not block legitimate requests
    except Exception as e:
        print(f"Error checking if plant: {e}")
        import traceback
        traceback.print_exc()
        return True  # Default to True if check fails, to not block legitimate requests

def get_fertilizer_recommendation(plant_name, disease_name, is_healthy, lat, lon):
    if is_healthy:
        prompt = f"""You are an expert agronomist in Pune, India. Provide a "Preventative Maintenance Guide" for a HEALTHY {plant_name} plant. 
        Include: 1. Organic immunity booster w/ dosage. 2. NPK ratio. 3. Water/soil tip. Keep it brief (80 words), bullet points, NO bold text."""
    else:
        prompt = f"""You are an expert agronomist in Pune, India. Provide a "Curative Treatment Plan" for {plant_name} with {disease_name}. 
        Include: 1. Chemical/fungicide name (Indian trade names). 2. Exact dosage. 3. Recovery fertilizer. Keep it brief (80 words), bullet points, NO bold text."""
    return ask_gemini(prompt)


@app.route('/chat', methods=['POST'])
def chat_route():
    user_message = request.json.get('message')
    if not user_message: return jsonify({'reply': "Please say something!"})
    
    prompt = f"""
    You are 'CropHealth Bot', a friendly AI farming assistant.
    User Question: "{user_message}"
    
    Instructions:
    - Answer ONLY agricultural questions (crops, soil, weather, fertilizers, diseases).
    - If the user asks about anything else (coding, movies, etc.), politely refuse.
    - Keep answers concise, helpful, and encouraging.
    """
    
    reply = ask_gemini(prompt)
    return jsonify({'reply': reply})


@app.route('/predict_disease', methods=['POST'])
def predict_disease_route():
    if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    lat = request.form.get('lat')
    lon = request.form.get('lon')

    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    if not disease_model: return jsonify({'error': 'Disease model is not loaded.'}), 500
    
    try:
        # Read the file stream into memory
        file.stream.seek(0)  # Reset stream position
        img = Image.open(file.stream).convert('RGB')
        
        # First, check with Gemini if it's a valid plant
        is_plant = check_if_plant(img.copy())
        
        if not is_plant:
            return jsonify({
                'disease': 'Not a Plant',
                'confidence': '0.00',
                'is_healthy': False,
                'prescription': '⚠️ Invalid Image',
                'fertilizer': 'The uploaded image does not appear to contain a valid plant, leaf, or plant part. Please upload an image of a plant leaf or crop for disease detection.',
                'not_plant': True
            })
        
        # If it's a plant, proceed with model prediction
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        prediction = disease_model.predict(img_array)
        pred_index = np.argmax(prediction[0])
        pred_class_name = DISEASE_CLASS_NAMES[pred_index]
        confidence = np.max(prediction[0]) * 100
        
        display_name = pred_class_name.replace('___', ' - ').replace('_', ' ')
        is_healthy = 'healthy' in pred_class_name.lower()
        plant_name = pred_class_name.split('___')[0]
        disease_name = display_name.split(' - ')[-1]

        fertilizer_rec = get_fertilizer_recommendation(plant_name, disease_name, is_healthy, lat, lon)
        prescription_title = "🛡️ Prevention & Maintenance Guide" if is_healthy else "💊 Curative Treatment Plan"

        return jsonify({
            'disease': display_name, 
            'confidence': f"{confidence:.2f}",
            'is_healthy': is_healthy,
            'prescription': prescription_title,
            'fertilizer': fertilizer_rec,
            'not_plant': False
        })
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': f'Analysis failed: {e}'}), 500


@app.route('/')
def home(): return render_template('index.html')
@app.route('/scanner')
def scanner(): return render_template('vision_checker.html')
@app.route('/library')
def library(): return render_template('library.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
