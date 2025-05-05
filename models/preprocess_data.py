import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
file_path = "data/crop_drought_resistance_prediction.csv"
data = pd.read_csv(file_path)
print(f"Data loaded successfully. Shape: {data.shape}")

# Handle missing values by filling with the mean
data = data.fillna(data.mean())
print("Missing values handled.")

# Remove outliers using Z-score
z_scores = np.abs((data - data.mean()) / data.std())
data = data[(z_scores < 3).all(axis=1)]
print("Outliers removed.")

# Normalize numerical features
feature_columns = [col for col in data.columns if col != "Target"]  # Exclude target column
scaler = StandardScaler()
data[feature_columns] = scaler.fit_transform(data[feature_columns])
print("Features normalized.")

# Save the preprocessed data
output_file = "../data/crop_drought_resistance_data.csv"
data.to_csv(output_file, index=False)
print(f"Preprocessed data saved to {output_file}")

# Save the scaler for later use
scaler_file = "models/scaler.pkl"
joblib.dump(scaler, scaler_file)
print(f"Scaler saved to {scaler_file}")