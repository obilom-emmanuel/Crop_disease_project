import tensorflow as tf
import numpy as np
import os
import json
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 226
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

DATASET_PATH = r"C:\Users\ekenelovedayobilom\Downloads\PlantVillage Dataset\plantvillage dataset\color"

MODEL_PATH = "crop_disease_model.h5"
CLASS_NAMES_PATH = "class_names.json"


# ============================================================
# FIND DATASET
# ============================================================

dataset_path = Path(DATASET_PATH)

if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset folder was not found:\n{DATASET_PATH}"
    )

print("\nDataset folder found:")
print(dataset_path)


# ============================================================
# FIND MAIZE CLASSES
# ============================================================

maize_folders = sorted(
    [
        folder
        for folder in dataset_path.iterdir()
        if folder.is_dir()
        and folder.name.startswith("Corn_(maize)")
    ]
)

if not maize_folders:
    raise FileNotFoundError(
        "No Corn_(maize) folders were found."
    )


print("\n============================================")
print("MAIZE CLASSES FOUND")
print("============================================")

for folder in maize_folders:
    print(folder.name)

print("\nNumber of maize classes:", len(maize_folders))


# ============================================================
# CREATE CLASS NAMES
# ============================================================

class_names = [
    folder.name
    for folder in maize_folders
]

class_to_index = {
    name: index
    for index, name in enumerate(class_names)
}


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

    class_images = []

    for image_file in folder.iterdir():

        if image_file.is_file() and image_file.suffix in valid_extensions:

            class_images.append(str(image_file))

    print(
        f"{class_name}: {len(class_images)} images"
    )

    for image_path in class_images:

        image_paths.append(image_path)
        labels.append(class_index)


print("\n============================================")
print("DATASET INFORMATION")
print("============================================")

print(
    "Total maize images:",
    len(image_paths)
)


# ============================================================
# CONVERT TO NUMPY ARRAYS
# ============================================================

image_paths = np.array(image_paths)
labels = np.array(labels)


# ============================================================
# SHUFFLE DATA
# ============================================================

rng = np.random.default_rng(SEED)

indices = rng.permutation(
    len(image_paths)
)

image_paths = image_paths[indices]
labels = labels[indices]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(image_paths) * 0.80
)

train_paths = image_paths[:split_index]
train_labels = labels[:split_index]

validation_paths = image_paths[split_index:]
validation_labels = labels[split_index:]


print(
    "\nTraining images:",
    len(train_paths)
)

print(
    "Validation images:",
    len(validation_paths)
)


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

    image.set_shape(
        [None, None, 3]
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    return image, label


# ============================================================
# CREATE TRAINING DATASET
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_paths,
        train_labels
    )
)

train_dataset = (
    train_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .shuffle(
        1000,
        seed=SEED
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# CREATE VALIDATION DATASET
# ============================================================

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (
        validation_paths,
        validation_labels
    )
)

validation_dataset = (
    validation_dataset
    .map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.1
        ),

        tf.keras.layers.RandomZoom(
            0.1
        )
    ],
    name="data_augmentation"
)


# ============================================================
# BUILD CNN MODEL
# ============================================================

model = tf.keras.Sequential(
    [

        tf.keras.layers.Input(
            shape=(
                IMG_SIZE,
                IMG_SIZE,
                3
            )
        ),

        # Data augmentation
        data_augmentation,

        # Normalize pixel values
        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

        # ----------------------------------------
        # CONVOLUTIONAL BLOCK 1
        # ----------------------------------------

        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------
        # CONVOLUTIONAL BLOCK 2
        # ----------------------------------------

        tf.keras.layers.Conv2D(
            64,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------
        # CONVOLUTIONAL BLOCK 3
        # ----------------------------------------

        tf.keras.layers.Conv2D(
            128,
            (3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            (2, 2)
        ),

        # ----------------------------------------
        # CLASSIFICATION HEAD
        # ----------------------------------------

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            len(class_names),
            activation="softmax"
        )
    ]
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\n============================================")
print("CNN MODEL")
print("============================================")

model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n============================================")
print("STARTING MAIZE DISEASE MODEL TRAINING")
print("============================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n============================================")
print("MODEL EVALUATION")
print("============================================")

validation_loss, validation_accuracy = model.evaluate(
    validation_dataset
)

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

model.save(
    MODEL_PATH
)

print(
    f"Model saved successfully as: {MODEL_PATH}"
)


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


print(
    f"Class names saved as: {CLASS_NAMES_PATH}"
)


# ============================================================
# DISPLAY FINAL CLASS MAPPING
# ============================================================

print("\n============================================")
print("FINAL CLASS MAPPING")
print("============================================")

for index, class_name in enumerate(class_names):

    print(
        f"{index} -> {class_name}"
    )


print("\n============================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("============================================")