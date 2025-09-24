import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

def train_vision_model():
    """
    Handles the entire process of training the image recognition model.
    """
    print("Starting the vision model training process...")

    base_dir = 'New Plant Diseases Dataset(Augmented)' 
    
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'valid')

    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        print("\n--- ERROR ---")
        print(f"Could not find the 'train' and 'valid' directories inside '{base_dir}'.")
        print("Please make sure you have unzipped the dataset correctly and the path is correct.")
        print("---------------\n")
        return

    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    BATCH_SIZE = 32 
    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'  
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Found {train_generator.samples} training images belonging to {num_classes} classes.")
    print(f"Found {validation_generator.samples} validation images.")

    print("Building the Convolutional Neural Network (CNN)...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(512, activation='relu'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary() 

    print("\n--- Starting Model Training ---")
    print("This will take a significant amount of time. Please be patient.")
    
    EPOCHS = 10

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE
    )

    print("--- Model Training Complete! ---")
    model_filename = 'plant_disease_model.h5'
    model.save(model_filename)
    print(f"\nModel saved successfully as '{model_filename}'")

if __name__ == "__main__":
    train_vision_model()