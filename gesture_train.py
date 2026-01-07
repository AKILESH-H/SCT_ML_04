import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\AKILESH\Desktop\Akilesh\internship\SCT_ML_4\archive\leapGestRecog"
IMG_DIM = 64
EPOCHS = 12
BATCH_SIZE = 32

X_data, y_data = [], []
gesture_names = []

print("Loading images...")

for subject_folder in os.listdir(DATA_DIR):
    subject_path = os.path.join(DATA_DIR, subject_folder)
    if not os.path.isdir(subject_path):
        continue

    for gesture_folder in os.listdir(subject_path):
        gesture_path = os.path.join(subject_path, gesture_folder)
        if not os.path.isdir(gesture_path):
            continue

        if gesture_folder not in gesture_names:
            gesture_names.append(gesture_folder)
        label_index = gesture_names.index(gesture_folder)

        for img_file in os.listdir(gesture_path):
            img_path = os.path.join(gesture_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, (IMG_DIM, IMG_DIM))
            X_data.append(img_resized)
            y_data.append(label_index)

if len(X_data) == 0:
    print("No images found! Check your DATA_DIR.")
    exit()

X_data = np.array(X_data).reshape(-1, IMG_DIM, IMG_DIM, 1) / 255.0
y_data = np.array(y_data)

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# ---------------- CNN Model ----------------
model = Sequential([
    Input(shape=(IMG_DIM, IMG_DIM, 1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(gesture_names), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ---------------- Save Model ----------------
model.save("gesture_cnn_model.h5")
with open("gesture_labels.txt", "w") as f:
    for gesture in gesture_names:
        f.write(gesture + "\n")

print("\nModel saved as gesture_cnn_model.h5")
print("Gesture labels saved as gesture_labels.txt")

# ---------------- Plot ----------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title("Loss")
plt.legend()
plt.show()
