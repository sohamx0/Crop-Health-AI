import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import warnings

# Suppress unnecessary TensorFlow warnings for a cleaner output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def predict_plant_disease():
    """
    Loads the trained vision model and predicts the disease for a single image.
    This is a standalone script for developer testing.
    """
    print("--- AI Plant Disease Diagnosis Tool (Developer Mode) ---")

    # --- Configuration ---
    model_path = 'plant_disease_model.h5'
    
    # This is a hardcoded list of class names that MUST match the order from training.
    # This makes the script independent of the original dataset folder.
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
        'Tomato___healthy'
    ]

    # --- IMPORTANT: PASTE THE FULL PATH TO YOUR IMAGE HERE ---
    # Use an 'r' before the string to handle backslashes correctly on Windows.
    image_path = r'PASTE_YOUR_IMAGE_PATH_HERE' # <--- CHANGE THIS

    # --- Model Loading ---
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file not found at '{model_path}'")
        return
    
    print(f"\nLoading model from: {model_path}...")
    try:
        model = load_model(model_path, compile=False) # Use compile=False for faster loading
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"\n[ERROR] Failed to load the model: {e}")
        return

    # --- Image Validation and Prediction ---
    if 'PASTE_YOUR_IMAGE_PATH_HERE' in image_path or not os.path.exists(image_path):
        print(f"\n[ERROR] Invalid image path. Please paste a valid file path on line 42.")
        return
        
    try:
        print(f"\nAnalyzing image: {os.path.basename(image_path)}")
        
        # Load and preprocess the image
        img = image.load_img(image_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        
        # Get the top prediction
        predicted_class_index = np.argmax(predictions[0])
        confidence = round(100 * np.max(predictions[0]), 2)
        
        if predicted_class_index >= len(class_names):
            print("\n[ERROR] Prediction index is out of bounds. The class name list might be wrong.")
            return

        predicted_class_name = class_names[predicted_class_index]
        # Clean up the name for display
        display_name = predicted_class_name.replace('___', ' - ').replace('_', ' ')

        # --- Display Result ---
        print("\n--- AI Diagnosis ---")
        print(f"Predicted Disease: {display_name}")
        print(f"Confidence: {confidence}%")
        print("--------------------")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during prediction: {e}")


if __name__ == '__main__':
    predict_plant_disease()