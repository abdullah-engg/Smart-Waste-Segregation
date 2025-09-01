import os
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# -----------------------------
# 1. Dataset paths
# -----------------------------
DATASET_PATH = r"C:\D disc\proto1\dataset"
CATEGORIES = ["bottle", "can", "paper"]  # class names

# -----------------------------
# 2. Parameters
# -----------------------------
SAMPLE_RATE = 22050  # Hz
DURATION = 50  # seconds per sample
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
N_MELS = 64  # Mel bands

# -----------------------------
# 3. Feature extraction
# -----------------------------
def extract_features(file_path):
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pad or truncate to fixed length
    if len(signal) > SAMPLES_PER_FILE:
        signal = signal[:SAMPLES_PER_FILE]
    else:
        signal = np.pad(signal, (0, max(0, SAMPLES_PER_FILE - len(signal))))

    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db

# -----------------------------
# 4. Load dataset
# -----------------------------
X, y = [], []

for idx, category in enumerate(CATEGORIES):
    folder = os.path.join(DATASET_PATH, category)
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path = os.path.join(folder, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(idx)

X = np.array(X)
y = np.array(y)

# Add channel dimension for CNN
X = X[..., np.newaxis]

# One-hot encode labels
y = to_categorical(y, num_classes=len(CATEGORIES))

# -----------------------------
# 5. Train/Test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6. CNN Model
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(CATEGORIES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 7. Train
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# -----------------------------
# 8. Evaluate
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# -----------------------------
# 9. Save model
# -----------------------------
model.save("audio_classifier.h5")
