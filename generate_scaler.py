import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Example training data (replace this with your actual training data)
training_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Fit the scaler
scaler = StandardScaler()
scaler.fit(training_data)

# Save the scaler to the models directory
models_dir = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(models_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(models_dir, "scaler.pkl"))

print("scaler.pkl has been generated and saved to the models directory.")