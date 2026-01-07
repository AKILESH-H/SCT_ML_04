import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

IMG_DIM = 64
MODEL_FILE = "gesture_cnn_model.h5"
LABEL_FILE = "gesture_labels.txt"

model = load_model(MODEL_FILE)
with open(LABEL_FILE, "r") as f:
    gestures = [line.strip() for line in f.readlines()]

Tk().withdraw()
file_path = filedialog.askopenfilename(title="Select a hand gesture image")

if not file_path:
    print("No image selected.")
    exit()

img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
img_resized = cv2.resize(img, (IMG_DIM, IMG_DIM))
img_input = img_resized.reshape(1, IMG_DIM, IMG_DIM, 1) / 255.0

pred = model.predict(img_input, verbose=0)
index = np.argmax(pred)
confidence = pred[0][index]*100
label = gestures[index]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap='gray')
plt.title(f"Predicted: {label}")
plt.axis('off')

plt.subplot(1,2,2)
plt.barh([label], [confidence], color='skyblue')
plt.xlim(0,100)
plt.title("Confidence (%)")
plt.show()
