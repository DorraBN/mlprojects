from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"

CHURN_FEATURES = ["tenure", "Contract", "InternetService", "MonthlyCharges"]
REGRESSION_FEATURES = ["tenure", "MonthlyCharges"]
SEGMENTATION_FEATURES = ["tenure", "MonthlyCharges"]


def load_dataset(path="dataset.csv"):
    """Read the dataset and normalize common Telco dataset columns."""
    df = pd.read_csv(path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean_dataset(df):
    """Remove rows without the target needed for the relevant training tasks."""
    df = df.copy()
    if "Churn" in df.columns:
        df = df.dropna(subset=["Churn"])
    return df


def make_preprocessor(numeric_features, categorical_features, scale_numeric=True):
    """Build a reusable preprocessing transformer fitted only on training data."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    transformers = [("num", numeric_pipeline, numeric_features)]
    if categorical_features:
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)


def validate_required_columns(df, required_columns):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(
            "The dataset is missing these required columns: " + ", ".join(missing)
        )
