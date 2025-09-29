import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

def predict_plant_disease():
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
    
    image_path = r'C:\Users\Lenovo\EDI\New Plant Diseases Dataset(Augmented)\train\Cherry_(including_sour)___healthy\0a0bd696-c093-47ef-866b-7f5a40af3edb___JR_HL 3952.JPG' # <--- CHANGE THIS FOR NEW IMAGES

    if 'PASTE_YOUR_IMAGE_PATH_HERE' in image_path or not os.path.exists(image_path):
        print("\n--- ACTION REQUIRED ---")
        print("Please open this script (predict_image.py) and change the 'image_path' variable")
        print("to the full path of an image you want to test.")
        print("You can find images inside the 'New Plant Diseases Dataset(Augmented)/valid' folder.")
        print("-----------------------\n")
        return

    img = image.load_img(image_path, target_size=(128, 128))
    
    img_array = image.img_to_array(img)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    img_array /= 255.0

    prediction = model.predict(img_array)

    predicted_class_index = np.argmax(prediction[0])
    
    train_dir = 'New Plant Diseases Dataset(Augmented)/train'
    class_names = sorted(os.listdir(train_dir))

    predicted_class_name = class_names[predicted_class_index]
    confidence = prediction[0][predicted_class_index] * 100

    print("\n--- AI Diagnosis ---")
    print(f"Image Path: {image_path}")
    print(f"==> Predicted Disease: {predicted_class_name.replace('___', ' ')}")
    print(f"Confidence: {confidence:.2f}%")
    print("--------------------\n")

if __name__ == "__main__":
    predict_plant_disease()