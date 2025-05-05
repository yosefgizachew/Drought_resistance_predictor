from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
MODELS_DIR = os.path.join(BASE_DIR, "../models")       # Adjust path to models directory

suggestion_model = load_model(os.path.join(MODELS_DIR, "suggestion_model.h5"))
prediction_model = load_model(os.path.join(MODELS_DIR, "prediction_model.h5"))

# Load scaler
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/suggestion')
def suggestion():
    return render_template('suggestion.html')

@app.route('/suggestion/result', methods=['POST'])
def suggestion_result():
    try:
        # Check if the request is form data
        if request.form:
            # Convert form data to JSON format
            data = [float(request.form[key]) for key in request.form.keys()]
        else:
            # Otherwise, handle JSON
            if not request.is_json:
                return jsonify({'error': "Unsupported Media Type: Content-Type must be 'application/json'"}), 415
            data = request.json.get('data')
            if not data or not isinstance(data, list):
                raise ValueError("Invalid input data. Expected a list of numbers.")

        # Validate input shape (16 features)
        if len(data) != 16:
            raise ValueError(f"Expected 16 features, but got {len(data)} features.")

        # Transform the input data
        data = scaler.transform([data])
        suggestion = suggestion_model.predict(data)

        # Check if the model output is a probability distribution
        if suggestion.ndim == 2 and suggestion.shape[1] > 1:
            # Multi-class output: Get the index of the highest probability
            hybrid_id = int(np.argmax(suggestion))
        else:
            # Single output: Assume it's already the hybrid ID
            hybrid_id = int(suggestion[0][0])

        # Render the result in an HTML template
        return render_template('result.html', result=f"Recommended Hybrid: {hybrid_id}")
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return render_template('error.html', error=str(e)), 500

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/prediction/result', methods=['POST'])
def prediction_result():
    try:
        # Get user input
        data = [float(request.form[key]) for key in request.form.keys()]
        data = scaler.transform([data])
        
        # Predict suitability
        suitability = prediction_model.predict(data)[0][0]
        
        return render_template('result.html', result=f"Suitability Probability: {suitability:.2f}")
    except Exception as e:
        app.logger.error(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)