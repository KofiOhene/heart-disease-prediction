import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_logistic_regression(train_features, train_labels):
    """
    Trains a Logistic Regression model on the training dataset.
    """
    print("\nTraining Logistic Regression Model...")

    # Initialize and train model
    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(train_features, train_labels)

    print("Model Training Complete!")

    return model


def evaluate_model(model, test_features, test_labels):
    """
    Evaluates the model on test data.
    """
    print("Evaluating Model...")

    # Make predictions
    predictions = model.predict(test_features)

    # Model metrics
    accuracy = accuracy_score(test_labels, predictions)
    conf_matrix = confusion_matrix(test_labels, predictions)
    class_report = classification_report(test_labels, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    return predictions


def save_model(model, filename="models/logistic_regression.pkl"):
    """
    Saves the trained model, creating the directory if necessary.
    """
    model_dir = os.path.dirname(filename)

    # Ensure directory exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")

    # Save the model
    joblib.dump(model, filename)
    print(f"\nModel saved successfully at: {filename}")
