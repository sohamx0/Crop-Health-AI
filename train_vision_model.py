# train_vision_model.py

# Step 1: Make sure you've installed the necessary libraries
# pip install tensorflow
# pip install numpy
# pip install matplotlib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# --- What this script does ---
# 1. Locates the massive image dataset you downloaded.
# 2. Sets up an "ImageDataGenerator" to load, resize, and prepare the images for training.
# 3. Builds a Convolutional Neural Network (CNN) - a special type of AI for understanding images.
# 4. Trains the CNN on the plant disease images. This is the time-consuming part.
# 5. Saves the final trained model to a file named `plant_disease_model.h5`.

def train_vision_model():
    """
    Handles the entire process of training the image recognition model.
    """
    print("Starting the vision model training process...")

    # --- Step 2: Set up paths and parameters ---
    # IMPORTANT: Update this path to where you unzipped the dataset.
    # It should point to the folder that contains 'train' and 'valid' subfolders.
    base_dir = 'New Plant Diseases Dataset(Augmented)' # <--- CHANGE THIS IF NEEDED
    
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'valid')

    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        print("\n--- ERROR ---")
        print(f"Could not find the 'train' and 'valid' directories inside '{base_dir}'.")
        print("Please make sure you have unzipped the dataset correctly and the path is correct.")
        print("---------------\n")
        return

    # Image parameters
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32 # How many images to process at a time

    # --- Step 3: Prepare the Data (Image Augmentation) ---
    # We use ImageDataGenerator to automatically load images from the folders.
    # It also "augments" the data by randomly flipping, rotating, and zooming images.
    # This makes our model more robust and prevents it from just memorizing the pictures.
    
    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255, # Normalize pixel values to be between 0 and 1
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # The validation data should not be augmented, only rescaled.
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Create the generators from the directories
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical' # For multiple disease categories
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    # Get the number of classes (disease categories)
    num_classes = len(train_generator.class_indices)
    print(f"Found {train_generator.samples} training images belonging to {num_classes} classes.")
    print(f"Found {validation_generator.samples} validation images.")


    # --- Step 4: Build the CNN Model ---
    print("Building the Convolutional Neural Network (CNN)...")
    model = Sequential([
        # 1st Convolutional Layer: Learns basic features like edges and corners
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),

        # 2nd Convolutional Layer: Learns more complex features
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # 3rd Convolutional Layer
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Flatten the results to feed into a dense network
        Flatten(),

        # A Dense (fully connected) layer for classification
        Dense(512, activation='relu'),
        
        # Dropout layer to prevent overfitting
        Dropout(0.5),

        # Output layer: has one neuron for each disease class
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary() # Print a summary of the model architecture

    # --- Step 5: Train the Model ---
    print("\n--- Starting Model Training ---")
    print("This will take a significant amount of time. Please be patient.")
    
    # We'll train for a few epochs. An epoch is one full pass through the entire dataset.
    EPOCHS = 10 # Start with 10, can increase for better accuracy

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    print("--- Model Training Complete! ---")

    # --- Step 6: Save the Model ---
    model_filename = 'plant_disease_model.h5'
    model.save(model_filename)
    print(f"\nModel saved successfully as '{model_filename}'")


# --- Main execution block ---
if __name__ == "__main__":
    train_vision_model()

