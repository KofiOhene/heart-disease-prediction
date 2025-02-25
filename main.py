import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_data
from src.model_training import train_logistic_regression, evaluate_model, save_model

#  Load the preprocessed dataset
file_path = "data/processed.cleveland.data"
df = load_data(file_path)

#  Define Features (train_features) and Target (train_labels)
train_features = df.drop(columns=["num"])  # Drop target column
train_labels = df["num"].apply(lambda x: 1 if x > 0 else 0)  # Convert to binary classification

#  Train-Test Split
train_features, test_features, train_labels, test_labels = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

#  Train the Logistic Regression Model
model = train_logistic_regression(train_features, train_labels)

# Evaluate the Model
evaluate_model(model, test_features, test_labels)

# Save the Model
save_model(model)
