# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
df = pd.read_csv("data/emotions.csv")
print("✅ Dataset loaded:", df.shape)

# Label encode the emotions (safe / alert / harmful)
encoder = LabelEncoder()
df["label"] = encoder.fit_transform(df["label"])  # example: 0=neutral, 1=happy, 2=stressed

# Split X and y
X = df.drop("label", axis=1).values
y = df["label"].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Build Model
# -------------------------------------------------------
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 emotion/stress levels
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2, verbose=1)

# -------------------------------------------------------
# Save Model and Preprocessing Tools
# -------------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/neuro_firewall_real.h5")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("✅ Model trained and saved successfully!")
