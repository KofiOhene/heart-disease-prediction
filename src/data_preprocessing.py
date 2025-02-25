import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder


def scale_features(df, columns_to_scale):
    """
    Standardize numerical features (Z-score normalization).
    """
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    print("\n✅ Scaled Numerical Features:\n", df[columns_to_scale].head())
    return df


def encode_features(df, one_hot_columns, label_columns):
    """
    Encode categorical features using One-Hot Encoding and Label Encoding.
    """
    # 🔹 One-Hot Encoding
    df = pd.get_dummies(df, columns=one_hot_columns, drop_first=True)

    # 🔹 Label Encoding for binary categorical variables
    label_encoder = LabelEncoder()
    for col in label_columns:
        df[col] = label_encoder.fit_transform(df[col])

    print("\n✅ Encoded Categorical Features:\n", df.head())
    return df


def preprocess_features(df):
    """
    Applies feature scaling and encoding to the dataset.
    """
    # ✅ Step 1: Define columns to scale & encode
    numerical_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    one_hot_features = ["cp", "restecg", "slope", "thal"]
    label_features = ["sex", "fbs", "exang", "num"]

    # ✅ Step 2: Apply Scaling
    df = scale_features(df, numerical_features)

    # ✅ Step 3: Apply Encoding
    df = encode_features(df, one_hot_features, label_features)

    return df
