import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

train_dir = 'path/to/rice_disease_dataset/train'
test_dir = 'path/to/rice_disease_dataset/test'
val_dir = 'path/to/rice_disease_dataset/val'

def apply_eclahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eclahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return eclahe.apply(gray)

def kgdc_segmentation(image):
    segmented_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)[1]
    return segmented_image

base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

def extract_features(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(img_to_array(image), axis=0) / 255.0
    features = base_model.predict(image)
    return features.flatten()

def optimize_features(features):
    optimized_features = features  # Replace with PSAO logic if available
    return optimized_features

def process_dataset(data_dir):
    X, y = [], []
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            image = load_img(image_path)
            image = img_to_array(image)
            eclahe_image = apply_eclahe(image)
            segmented_image = kgdc_segmentation(eclahe_image)
            features = extract_features(segmented_image)
            optimized_features = optimize_features(features)
            X.append(optimized_features)
            y.append(label)
    return np.array(X), np.array(y)

X_train, y_train = process_dataset(train_dir)
X_test, y_test = process_dataset(test_dir)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
