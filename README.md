Heart Disease Prediction API

This project is a machine learning-based API that predicts heart disease risk using patient data. The API is built with Flask and serves a logistic regression model trained on the Cleveland heart disease dataset.

heartdisease/
│── app/
│   └── app.py                # Flask API for model deployment
│
│── data/
│   ├── processed.cleveland.data  # Raw dataset
│   ├── X_train.csv           # Training data (features)
│   ├── X_test.csv            # Testing data (features)
│   ├── y_train.csv           # Training data (labels)
│   ├── y_test.csv            # Testing data (labels)
│
│── models/
│   └── logistic_regression.pkl  # Trained model
│
│── src/
│   ├── data_exploration.py   # Exploratory Data Analysis (EDA)
│   ├── data_loader.py        # Loads and cleans dataset
│   ├── data_preprocessing.py # Feature scaling & encoding
│   ├── model_training.py     # Model training and evaluation
│
│── venv/                     # Virtual environment
│── main.py                    # Main script for data processing & training
│── requirements.txt            # List of dependencies
│── README.md                   # Project documentation


Model Performance


The logistic regression model achieved:

Accuracy: 85.25%
Precision (Class 0): 0.90
Recall (Class 1): 0.89






