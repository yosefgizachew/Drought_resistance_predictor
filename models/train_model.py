import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import joblib

# Load datasets
suggestion_data = pd.read_csv('../data-sets/crop_drought_resistance_suggestion.csv')  # Dataset for hybrid suggestion
prediction_data = pd.read_csv('../data-sets/crop_drought_resistance_prediction.csv')  # Dataset for suitability prediction

# Preprocessing for Suggestion Dataset
suggestion_features = [
    'Precipitation_mm', 'Temperature_C', 'Solar_Radiation_MJ_m2', 'Soil_Moisture_%',
    'Humidity_%', 'Drought_Duration_days', 'WUE_g_per_mm', 'Leaf_Water_Potential_MPa',
    'Stomatal_Conductance_mol_m2_s', 'Root_Depth_cm', 'Photosynthetic_Rate_umol_m2_s',
    'Plant_Biomass_g_m2', 'ZmDREB2A', 'Root_QTL', 'ZmNAC', 'Planting_Density_plants_ha'
]
X_suggestion = suggestion_data[suggestion_features]
y_suggestion = pd.get_dummies(suggestion_data['Hybrid_ID'])  # One-hot encoding for multi-class output

# Standardize features
scaler = StandardScaler()
X_suggestion = scaler.fit_transform(X_suggestion)

# Train-test split
X_train_suggestion, X_test_suggestion, y_train_suggestion, y_test_suggestion = train_test_split(
    X_suggestion, y_suggestion, test_size=0.2, random_state=42
)

# Build Neural Network for Suggestion
suggestion_model = Sequential([
    Dense(128, input_dim=X_suggestion.shape[1], activation='relu'),  # Input layer
    Dropout(0.2),
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.2),
    Dense(y_suggestion.shape[1], activation='softmax')  # Output layer (softmax for multi-class)
])

suggestion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
suggestion_history = suggestion_model.fit(
    X_train_suggestion, y_train_suggestion,
    validation_data=(X_test_suggestion, y_test_suggestion),
    epochs=50, batch_size=32, callbacks=[early_stopping]
)

# Evaluate the model
_, suggestion_accuracy = suggestion_model.evaluate(X_test_suggestion, y_test_suggestion)
print(f'Suggestion Model Accuracy: {suggestion_accuracy:.3f}')

# Save the model


# Preprocessing for Prediction Dataset
# Use the same features as suggestion but without 'Hybrid_ID'
prediction_features = suggestion_features
X_prediction = prediction_data[prediction_features]
y_prediction = prediction_data['Drought_Score']  # Use 'Drought_Score' as the target

# Standardize features
X_prediction = scaler.transform(X_prediction)

# Train-test split
X_train_prediction, X_test_prediction, y_train_prediction, y_test_prediction = train_test_split(
    X_prediction, y_prediction, test_size=0.2, random_state=42
)

# Build Neural Network for Prediction
prediction_model = Sequential([
    Dense(128, input_dim=X_prediction.shape[1], activation='relu'),  # Input layer
    Dropout(0.2),
    Dense(64, activation='relu'),  # Hidden layer
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer (sigmoid for binary classification)
])

prediction_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with early stopping
prediction_history = prediction_model.fit(
    X_train_prediction, y_train_prediction,
    validation_data=(X_test_prediction, y_test_prediction),
    epochs=50, batch_size=32, callbacks=[early_stopping]
)

# Evaluate the model
_, prediction_accuracy = prediction_model.evaluate(X_test_prediction, y_test_prediction)
print(f'Prediction Model Accuracy: {prediction_accuracy:.3f}')

# Save the model
prediction_model.save('prediction_model.h5')

# Example Input for Hybrid Suggestion
suggestion_input = np.array([[400, 25, 20, 30, 60, 10, 3.5, -1.2, 0.4, 50, 25, 200, 1, 0, 1, 60000]])
suggestion_input = scaler.transform(suggestion_input)

# Predict Hybrid
suggested_hybrid_prob = suggestion_model.predict(suggestion_input)
suggested_hybrid = np.argmax(suggested_hybrid_prob)
print(f'Recommended Hybrid Index: {suggested_hybrid}')

# Example Input for Suitability Prediction
# Use the same input but exclude 'Hybrid_ID'
prediction_input = suggestion_input
suitability_prob = prediction_model.predict(prediction_input)
print(f'Suitability Probability: {suitability_prob[0][0]:.2f}')


# Plot training and validation loss
def plot_training_history(history, model_name="model"):
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_loss.png')  # Save the loss plot as an image
    plt.show()

    # Accuracy plot (if available)
    if 'accuracy' in history.history:
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
        plt.title(f'{model_name} Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{model_name}_accuracy.png')  # Save the accuracy plot as an image
        plt.show()

# Generate plots for the suggestion model
plot_training_history(suggestion_history, model_name="suggestion_model")
# Generate plots for the prediction model
plot_training_history(prediction_history, model_name="prediction_model")

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
