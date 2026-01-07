import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

IMG_DIM = 64
MODEL_FILE = "gesture_cnn_model.h5"
LABEL_FILE = "gesture_labels.txt"
DATASET_PATH = r"C:\Users\AKILESH\Desktop\Akilesh\internship\SCT_ML_4\archive\leapGestRecog"

model = load_model(MODEL_FILE)
with open(LABEL_FILE, "r") as f:
    gestures = [line.strip() for line in f.readlines()]

all_imgs = []
for sub in os.listdir(DATASET_PATH):
    sub_path = os.path.join(DATASET_PATH, sub)
    if not os.path.isdir(sub_path):
        continue
    for gest in os.listdir(sub_path):
        gest_path = os.path.join(sub_path, gest)
        if not os.path.isdir(gest_path):
            continue
        for img_file in os.listdir(gest_path):
            all_imgs.append(os.path.join(gest_path, img_file))

samples = random.sample(all_imgs, 4)
fig, axes = plt.subplots(4, 2, figsize=(12,16))

for i, img_path in enumerate(samples):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_DIM, IMG_DIM))
    img_input = img_resized.reshape(1, IMG_DIM, IMG_DIM, 1)/255.0

    pred = model.predict(img_input, verbose=0)
    index = np.argmax(pred)
    confidence = pred[0][index]*100
    label = gestures[index]

    axes[i][0].imshow(img, cmap='gray')
    axes[i][0].set_title(f"Predicted: {label}")
    axes[i][0].axis('off')

    axes[i][1].barh([label], [confidence], color='skyblue')
    axes[i][1].set_xlim(0,100)
    axes[i][1].set_title("Confidence (%)")

plt.tight_layout()
plt.show()
