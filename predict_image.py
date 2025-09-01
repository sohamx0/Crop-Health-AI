# predict_image.py

# Step 1: Make sure you have the necessary libraries
# pip install tensorflow
# pip install numpy
# pip install Pillow # For image manipulation

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# --- What this script does ---
# 1. Loads the big AI model we just trained (`plant_disease_model.h5`).
# 2. Loads a single image that you want to test.
# 3. Prepares the image to be the exact size and format the model expects.
# 4. Asks the model to predict the disease.
# 5. Prints the most likely diagnosis.

def predict_plant_disease():
    """
    Loads the trained model and makes a prediction on a single image.
    """
    # --- Step 2: Load the Trained Model ---
    model_path = 'plant_disease_model.h5'
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model '{model_path}' loaded successfully!")
    except Exception as e:
        print(f"\n--- ERROR ---")
        print(f"Could not load the model. Make sure '{model_path}' is in the same folder.")
        print(f"Error details: {e}")
        print("---------------\n")
        return

    # --- Step 3: Specify the Image to Test ---
    # IMPORTANT: You need to change this path to a real image file.
    # Go into the dataset folder, find an image in the 'valid' set, and copy its path here.
    # Use an 'r' before the string to prevent errors with backslashes on Windows.
    
    image_path = r'C:\Users\Lenovo\EDI\tomato-late-blight-tomato-1556463954.jpg' # <--- CHANGE THIS FOR NEW IMAGES

    # This check makes sure you've changed the placeholder text.
    if 'PASTE_YOUR_IMAGE_PATH_HERE' in image_path or not os.path.exists(image_path):
        print("\n--- ACTION REQUIRED ---")
        print("Please open this script (predict_image.py) and change the 'image_path' variable")
        print("to the full path of an image you want to test.")
        print("You can find images inside the 'New Plant Diseases Dataset(Augmented)/valid' folder.")
        print("-----------------------\n")
        return

    # --- Step 4: Load and Preprocess the Image ---
    # The model was trained on 128x128 images, so we need to resize our test image to match.
    img = image.load_img(image_path, target_size=(128, 128))
    
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    
    # The model expects a "batch" of images, so we add an extra dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize the pixel values just like we did for the training data
    img_array /= 255.0

    # --- Step 5: Make a Prediction ---
    prediction = model.predict(img_array)

    # The prediction is an array of probabilities for each class.
    # We find the one with the highest probability.
    predicted_class_index = np.argmax(prediction[0])
    
    # --- Step 6: Get the Class Labels ---
    # We need to know which index corresponds to which disease.
    # We can get this from the structure of the training directory.
    train_dir = 'New Plant Diseases Dataset(Augmented)/train'
    class_names = sorted(os.listdir(train_dir)) # Get class names from folder names

    # Get the name of the predicted class
    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    # --- Step 7: Display the Result ---
    print("\n--- AI Diagnosis ---")
    print(f"Image Path: {image_path}")
    print(f"==> Predicted Disease: {predicted_class_name.replace('___', ' ')}")
    print(f"Confidence: {confidence:.2f}%")
    print("--------------------\n")


# --- Main execution block ---
if __name__ == "__main__":
    predict_plant_disease()
