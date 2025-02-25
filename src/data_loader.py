import pandas as pd
from src.data_preprocessing import preprocess_features


def convert_numeric_columns(df):
    """
    Convert specific numeric columns, handling non-numeric values and ensuring proper formatting.
    """
    numeric_columns = ["chol", "fbs", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

    print("\n🔍 Before Conversion - Data Types:")
    print(df[numeric_columns].dtypes)

    # ✅ Step 1: Replace '?' with NaN
    df[numeric_columns] = df[numeric_columns].replace("?", None)

    # ✅ Step 2: Convert columns to numeric
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")

    # ✅ Step 3: Handle missing values (use median for numerical columns)
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    print("\n✅ After Conversion - Data Types:")
    print(df[numeric_columns].dtypes)

    return df


def load_data(file_path):
    """
    Load dataset from a .data file, clean, preprocess (scale & encode), and return as DataFrame.
    """
    # ✅ **Columns specific to the Cleveland dataset**
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

    print("\n🚀 Loading Dataset...")
    df = pd.read_csv(file_path, names=column_names, header=None)

    # ✅ **Convert necessary numeric columns**
    df = convert_numeric_columns(df)

    # ✅ **Apply Feature Scaling & Encoding**
    df = preprocess_features(df)

    print("\n✅ Final Processed Data Overview:")
    print(df.head())

    return df
