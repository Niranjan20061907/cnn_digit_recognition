import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ---------------------------------
# 1. Load sklearn digits dataset
# ---------------------------------
digits = load_digits()

X = digits.images  # (1797, 8, 8)
y = digits.target

# ---------------------------------
# 2. Preprocessing
# ---------------------------------
X = X / 16.0  # Normalize
X = X.reshape(-1, 8, 8, 1)  # CNN input shape
y = to_categorical(y, 10)  # One-hot labels

# ---------------------------------
# 3. Train-Test Split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------
# 4. Build CNN Model (Lightweight)
# ---------------------------------
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(8, 8, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# ---------------------------------
# 5. Compile Model
# ---------------------------------
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---------------------------------
# 6. Train Model
# ---------------------------------
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# ---------------------------------
# 7. SAVE MODEL
# ---------------------------------
model.save("sklearn_digits_cnn.h5")

print("✅ CNN model (sklearn digits) trained and saved")
