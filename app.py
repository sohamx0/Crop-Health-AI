from flask import Flask, render_template, request, jsonify
import requests
import json
from PIL import Image
import numpy as np
import os
import base64
import io
from typing import Any, Optional

try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications.resnet50 import preprocess_input
except Exception as e:
    load_model = None
    preprocess_input = None
    print(f"⚠️  Warning: TensorFlow/Keras not available: {e}")

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

app = Flask(__name__, static_folder='assets', static_url_path='/assets')

# Basic hardening / config
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", str(8 * 1024 * 1024)))  # 8MB
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DISEASE_MODEL_PATH = os.getenv("DISEASE_MODEL_PATH", "plant_disease_model.h5")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", "class_names.json")
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD", "0") == "1"


# Load API key from environment variable
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
if not GEMINI_API_KEY:
    print("⚠️  Warning: GEMINI_API_KEY not found in environment variables. Please set it in .env file")
    print("   The AI features (chat and plant validation) will not work without the API key.")
else:
    print(f"✅ Gemini API key loaded (length: {len(GEMINI_API_KEY)})")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}" if GEMINI_API_KEY else ""


def _load_class_names(path: str) -> Optional[list[str]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("class names JSON must be a list of strings")
        return data
    except FileNotFoundError:
        print(f"⚠️  Warning: Class names file not found: {path}")
        return None
    except Exception as e:
        print(f"⚠️  Warning: Failed to load class names from {path}: {e}")
        return None

DISEASE_CLASS_NAMES = _load_class_names(CLASS_NAMES_PATH)

if SKIP_MODEL_LOAD:
    print("ℹ️  SKIP_MODEL_LOAD=1 set; skipping model load")
    disease_model = None
else:
    try:
        if load_model is None:
            raise RuntimeError("TensorFlow/Keras is not installed or failed to import")

        disease_model = load_model(DISEASE_MODEL_PATH)
        print("✅ Local disease detection model loaded successfully!")

        if DISEASE_CLASS_NAMES is not None:
            try:
                num_outputs = int(getattr(disease_model, "output_shape", [None, None])[-1])
                if len(DISEASE_CLASS_NAMES) != num_outputs:
                    print(
                        "⚠️  Warning: Class name count does not match model outputs "
                        f"({len(DISEASE_CLASS_NAMES)} vs {num_outputs})."
                    )
            except Exception:
                pass
        else:
            print(
                "⚠️  Warning: DISEASE_CLASS_NAMES not loaded. "
                "Predictions will be blocked to avoid wrong label mapping."
            )
    except Exception as e:
        print(f"❌ Error loading local disease model: {e}")
        disease_model = None


def ask_gemini(prompt_text: str) -> str:
    if not GEMINI_API_KEY or not GEMINI_URL:
        return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in .env file"
    
    payload = { "contents": [{ "parts": [{"text": prompt_text}] }] }
    try:
        response = requests.post(GEMINI_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=30)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return "Error: AI server returned no candidates."
            content = (candidates[0].get("content") or {}).get("parts") or []
            if not content:
                return "Error: AI server returned empty content."
            text = content[0].get("text")
            return text if isinstance(text, str) and text.strip() else "Error: AI server returned empty text."
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
            data = response.json()
            candidates = data.get("candidates") or []
            text = ((candidates[0].get("content") or {}).get("parts") or [{}])[0].get("text", "") if candidates else ""
            result = str(text).strip().lower()

            # Prompt requests ONLY yes/no; be strict to avoid false positives.
            is_valid_plant = result.startswith("yes")
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
    loc = "Pune, India"
    if lat is not None and lon is not None:
        loc = f"near latitude {lat:.4f}, longitude {lon:.4f} (use best-effort regional context)"

    if is_healthy:
        prompt = (
            f"You are an expert agronomist. The user is {loc}. "
            f"Provide a 'Preventative Maintenance Guide' for a HEALTHY {plant_name} plant. "
            "Include: 1) Organic immunity booster + exact dosage, 2) Suggested NPK ratio, 3) Water/soil tip. "
            "Keep it brief (~80 words), bullet points, NO bold text."
        )
    else:
        prompt = (
            f"You are an expert agronomist. The user is {loc}. "
            f"Provide a 'Curative Treatment Plan' for {plant_name} with {disease_name}. "
            "Include: 1) Chemical/fungicide name (prefer India-available trade/common names) + exact dosage, "
            "2) Application frequency, 3) Recovery fertilizer. "
            "Keep it brief (~80 words), bullet points, NO bold text."
        )

    return ask_gemini(prompt)


def _allowed_image_filename(filename: str) -> bool:
    _, ext = os.path.splitext(filename or "")
    return ext.lower() in ALLOWED_IMAGE_EXTENSIONS


def _format_class_for_display(class_name: str) -> str:
    # Preserve dataset class naming (may include spaces), but make it human-readable.
    name = (class_name or "").replace("___", " - ").replace("_", " ")
    return " ".join(name.split()).strip()


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
    lat_raw = request.form.get('lat')
    lon_raw = request.form.get('lon')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not _allowed_image_filename(file.filename):
        return jsonify({'error': 'Unsupported file type. Please upload a JPG, PNG, or WebP image.'}), 400
    if not disease_model:
        return jsonify({'error': 'Disease model is not loaded.'}), 500
    if not DISEASE_CLASS_NAMES:
        return jsonify({'error': 'Class names are not configured (missing class_names.json).'}), 500

    lat = None
    lon = None
    try:
        if lat_raw is not None and lon_raw is not None:
            lat = float(lat_raw)
            lon = float(lon_raw)
    except ValueError:
        lat = None
        lon = None
    
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
        if preprocess_input is None:
            raise RuntimeError("TensorFlow/Keras preprocess_input is unavailable")
        img_array = preprocess_input(img_array)
        
        prediction = disease_model.predict(img_array)
        pred_index = np.argmax(prediction[0])
        pred_class_name = DISEASE_CLASS_NAMES[pred_index]
        confidence = np.max(prediction[0]) * 100
        
        display_name = _format_class_for_display(pred_class_name)
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


@app.errorhandler(413)
def request_entity_too_large(_):
    return jsonify({'error': 'File too large. Please upload an image under the size limit.'}), 413


@app.route('/healthz')
def healthz():
    return jsonify(
        {
            'status': 'ok',
            'model_loaded': disease_model is not None,
            'class_names_loaded': bool(DISEASE_CLASS_NAMES),
            'gemini_configured': bool(GEMINI_API_KEY),
        }
    )


@app.route('/')
def home(): return render_template('index.html')
@app.route('/scanner')
def scanner(): return render_template('vision_checker.html')
@app.route('/library')
def library(): return render_template('library.html')

if __name__ == '__main__':
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)
