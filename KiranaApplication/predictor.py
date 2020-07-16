import tensorflow as tf
import cv2
from keras.models import load_model
import numpy as np

CATEGORIES = ["3Roses", "Hamam" ]

def prepare(filepath):
    IMG_SIZE = 50  
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

def predict():
    model = tf.keras.models.load_model("product-classifier.model")
    prediction = model.predict([prepare(r'image.jpeg')])
    return {
        "prediction":prediction,
        "category": CATEGORIES[int(prediction[0][0])]
    }
