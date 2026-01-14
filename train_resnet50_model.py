"""Train a ResNet50 transfer-learning model and write a matching class_names.json.

This script is optional (training can take a long time).
Dataset layout expected:
  New Plant Diseases Dataset(Augmented)/
    train/<class_name>/*.jpg
    valid/<class_name>/*.jpg

Outputs:
  - plant_disease_model.h5
  - class_names.json (index order matches model outputs)

Run:
  python train_resnet50_model.py
"""

import json
import os

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_resnet50_model():
    base_dir = os.getenv("DATASET_DIR", "New Plant Diseases Dataset(Augmented)")
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")

    if not os.path.exists(train_dir) or not os.path.exists(valid_dir):
        raise FileNotFoundError(
            f"Could not find dataset folders: {train_dir} and {valid_dir}. "
            "Set DATASET_DIR env var if your dataset path is different."
        )

    img_size = (224, 224)
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    epochs = int(os.getenv("EPOCHS", "5"))

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )
    valid_gen = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = len(train_gen.class_indices)
    print(f"Found {train_gen.samples} train images across {num_classes} classes")

    # Persist class index order used by the generator/model
    index_to_class = {index: name for name, index in train_gen.class_indices.items()}
    class_names = [index_to_class[i] for i in range(num_classes)]

    with open("class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print("Wrote class_names.json")

    base = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
    )

    model.save("plant_disease_model.h5")
    print("Saved plant_disease_model.h5")


if __name__ == "__main__":
    train_resnet50_model()
