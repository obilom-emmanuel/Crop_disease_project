import tensorflow as tf
import numpy as np
import os
import json


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 226

MODEL_FILE = "crop_disease_model.h5"
CLASS_NAMES_FILE = "class_names.json"


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names():
    """Load the class names used during model training."""

    if not os.path.exists(CLASS_NAMES_FILE):
        raise FileNotFoundError(
            f"{CLASS_NAMES_FILE} was not found."
        )

    with open(
        CLASS_NAMES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# BUILD CNN MODEL
# ============================================================

def build_cnn_model(num_classes):
    """
    Defines the CNN architecture used for maize
    disease classification.
    """

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(IMG_SIZE, IMG_SIZE, 3)
        ),

        tf.keras.layers.Rescaling(
            1.0 / 255
        ),

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

        # Classification Head
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(
            0.5
        ),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model


# ============================================================
# PREDICT MAIZE LEAF DISEASE
# ============================================================

def predict_leaf_disease(model_path, image_path):
    """
    Loads the trained maize disease model and predicts
    the disease present in a single leaf image.
    """

    # Check model
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    # Check image
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    # Load class names
    class_names = load_class_names()

    # Load trained model
    model = tf.keras.models.load_model(
        model_path
    )

    # Load image
    img = tf.keras.utils.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    # Convert image to array
    img_array = tf.keras.utils.img_to_array(
        img
    )

    # Create batch dimension
    img_array = tf.expand_dims(
        img_array,
        axis=0
    )

    # Make prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )

    # Get probabilities
    probabilities = predictions[0]

    # Get highest probability
    predicted_index = np.argmax(
        probabilities
    )

    confidence = (
        float(probabilities[predicted_index])
        * 100
    )

    predicted_class = class_names[
        predicted_index
    ]

    # Make the class name easier to read
    predicted_class = predicted_class.replace(
        "Corn_(maize)___",
        ""
    )

    predicted_class = predicted_class.replace(
        "_",
        " "
    )

    return predicted_class, confidence