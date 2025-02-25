from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("../models/logistic_regression.pkl")

# Define expected feature order based on training data
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope_1.0", "slope_2.0",
    "slope_3.0", "ca_0.0", "ca_1.0", "ca_2.0", "ca_3.0",
    "thal_3.0", "thal_6.0", "thal_7.0"
]


@app.route("/")
def home():
    return jsonify({"message": "Heart Disease Prediction API is running!"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON request
        data = request.json

        # Convert to DataFrame (ensure correct order of features)
        input_df = pd.DataFrame([data], columns=FEATURE_COLUMNS)

        # Convert to NumPy array for prediction
        input_array = input_df.to_numpy()

        # Make prediction
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0][1]  # Probability of heart disease

        return jsonify({
            "prediction": int(prediction),
            "probability": round(probability, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
