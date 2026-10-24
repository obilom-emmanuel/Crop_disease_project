import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 226
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = r"C:\Users\ekenelovedayobilom\Downloads\PlantVillage Dataset\plantvillage dataset\color"

MODEL_PATH = "crop_disease_model.h5"
CLASS_NAMES_PATH = "class_names.json"


# ============================================================
# FIND MAIZE CLASSES
# ============================================================

dataset_path = Path(DATASET_PATH)

maize_folders = sorted([
    folder for folder in dataset_path.iterdir()
    if folder.is_dir() and folder.name.startswith("Corn_(maize)")
])

if not maize_folders:
    raise FileNotFoundError(
        "No maize/corn folders were found. Check DATASET_PATH."
    )

print("\nMaize classes found:")

for folder in maize_folders:
    print(" -", folder.name)


# ============================================================
# CREATE CLASS NAMES
# ============================================================

class_names = [folder.name for folder in maize_folders]

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}

print("\nNumber of maize classes:", len(class_names))


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

image_paths = []
labels = []

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

for folder in maize_folders:

    class_name = folder.name
    class_index = class_to_index[class_name]

    for image_file in folder.iterdir():

        if image_file.suffix in valid_extensions:

            image_paths.append(str(image_file))
            labels.append(class_index)


print("\nTotal maize images found:", len(image_paths))


# ============================================================
# SHUFFLE DATA
# ============================================================

image_paths = np.array(image_paths)
labels = np.array(labels)

random_indices = np.random.permutation(len(image_paths))

image_paths = image_paths[random_indices]
labels = labels[random_indices]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(len(image_paths) * 0.8)

train_paths = image_paths[:split_index]
train_labels = labels[:split_index]

validation_paths = image_paths[split_index:]
validation_labels = labels[split_index:]

print("\nTraining images:", len(train_paths))
print("Validation images:", len(validation_paths))


# ============================================================
# IMAGE LOADING FUNCTION
# ============================================================

def load_image(path, label):

    image = tf.io.read_file(path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(image, tf.float32)

    return image, label


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_paths, train_labels)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (validation_paths, validation_labels)
)

train_dataset = (
    train_dataset
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])


# ============================================================
# BUILD CNN MODEL
# ============================================================

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    data_augmentation,

    tf.keras.layers.Rescaling(1.0 / 255),

    # Convolution Block 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Block 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    # Classification
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# DISPLAY MODEL STRUCTURE
# ============================================================

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nStarting maize disease model training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\n============================================")
print("MODEL TRAINING COMPLETED")
print("============================================")

print("Model saved as:", MODEL_PATH)


# ============================================================
# SAVE CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )

print("Class names saved as:", CLASS_NAMES_PATH)

print("\nMaize classes used by the model:")

for index, name in enumerate(class_names):

    print(
        index,
        "->",
        name
    )