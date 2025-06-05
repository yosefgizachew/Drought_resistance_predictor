# Crop Drought Resistance Prediction System

This project is a **Crop Drought Resistance Prediction System** that predicts the suitability of a crop hybrid for specific environmental conditions. It uses a **machine learning regression model** trained on environmental and crop data, and provides a user-friendly web interface for predictions.

---

## Features

* **Prediction Model**: Predicts the suitability score for a crop hybrid based on environmental features.
* **Web Interface**: A Flask-based web application for user interaction.
* **Error Handling**: Provides clear error messages for invalid inputs.
* **Scalable Deployment**: Includes Docker support for containerized deployment.

---

## Project Structure

```
project/
│
├── app/
│   ├── app.py                         # Flask application
│   ├── templates/
│   │   ├── index.html                 # Input form
│   │   ├── result.html                # Prediction result display
│   └── static/                        # Static files (CSS, JS, images)
│
├── scripts/
│   ├── preprocess_data.py            # Data preprocessing script
│   └── train_model.py                # Model training script
│
├── .gitignore
├── .dockerignore
├── Dockerfile                        # For Docker deployment
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## 1. Data Gathering

The dataset used for this project is stored at:
`data/crop_drought_resistance_prediction.csv`

### Dataset Contents

* **Features**: Environmental conditions (e.g., precipitation, temperature, soil moisture)
* **Target**: Suitability score for crop hybrids

### Steps

1. Collect data from reliable sources (e.g., public datasets, research institutions).
2. Ensure it contains:

   * Relevant environmental features
   * Suitability scores for crop hybrids
3. Save it as `crop_drought_resistance_prediction.csv` in the `data/` directory.

---

## 2. Model Training

The model is trained using a deep neural network algorithm via the `models/model.py` script.

### Steps

1. **Preprocess the data**:

   ```bash
   python models/preprocess_data.py
   ```

   * Normalizes numerical features using `StandardScaler`
   * Handles missing values and outliers

2. **Train the model**:

   ```bash
   python models/model.py
   ```

   * Saves the trained model to `models/prediction_model.h5`
   * Saves the scaler to `models/scaler.pkl`

---

## 3. Web Interface

The web interface is built using Flask and allows users to input environmental features for predictions.

### Steps to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app**:

   ```bash
   python app/app.py
   ```

3. Open a browser and go to:
   [http://localhost:5000](http://localhost:5000)

### Interface Features

* **Input Form**: Users provide 16 environmental features
* **Result Display**: Shows predicted suitability score
* **Error Handling**: Displays user-friendly error messages for invalid input

---

## 4. Deployment

A Dockerfile is included for containerized deployment.

### Steps to Deploy with Docker

1. **Build the Docker image**:

   ```bash
   docker build -t crop-prediction .
   ```

2. **Run the container**:

   ```bash
   docker run -p 5000:5000 crop-prediction
   ```

3. Visit the app at:
   [http://localhost:5000](http://localhost:5000)

---

## 5. Key Files and Scripts

### Key Files

* `app/app.py`: Flask application
* `models/prediction_model.h5`: Trained ML model
* `models/scaler.pkl`: Feature scaler
* `data/crop_drought_resistance_prediction.csv`: Training dataset

### Scripts

* `models/preprocess_data.py`: Preprocesses the dataset
* `models/train_model.py`: Trains and saves the model

---

## 6. Future Improvements

* Support multiple crop hybrids
* Add feature importance visualizations
* Deploy on a cloud platform (e.g., AWS, Heroku)

---

## 7. License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

