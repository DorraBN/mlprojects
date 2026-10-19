"""Train and save the final models used by the Streamlit application.

Run once (or whenever you want to retrain):
    python train_models.py
"""
import json
import warnings
import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import (
    BASE_DIR, MODEL_DIR, CHURN_FEATURES, REGRESSION_FEATURES,
    SEGMENTATION_FEATURES, load_dataset, clean_dataset,
    make_preprocessor, validate_required_columns
)

warnings.filterwarnings("ignore", category=UserWarning)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = clean_dataset(load_dataset())
required = list(set(CHURN_FEATURES + REGRESSION_FEATURES + ["Churn", "TotalCharges"]))
validate_required_columns(df, required)

# 1. Churn classification model + its preprocessing in one Pipeline.
churn_df = df.dropna(subset=CHURN_FEATURES + ["Churn"]).copy()
X_churn = churn_df[CHURN_FEATURES]
y_churn = churn_df["Churn"].astype(str).str.strip()

X_train, X_test, y_train, y_test = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)
numeric_churn = ["tenure", "MonthlyCharges"]
categorical_churn = ["Contract", "InternetService"]

churn_pipeline = Pipeline(steps=[
    ("preprocessor", make_preprocessor(numeric_churn, categorical_churn)),
    ("model", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )),
])
churn_pipeline.fit(X_train, y_train)
churn_pred = churn_pipeline.predict(X_test)

labels = sorted(y_churn.unique().tolist())
positive_label = "Yes" if "Yes" in labels else labels[-1]
metrics = {
    "classification": {
        "accuracy": float(accuracy_score(y_test, churn_pred)),
        "precision": float(precision_score(y_test, churn_pred, pos_label=positive_label, zero_division=0)),
        "recall": float(recall_score(y_test, churn_pred, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_test, churn_pred, pos_label=positive_label, zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, churn_pred, labels=labels).tolist(),
        "test_size": int(len(y_test)),
        "positive_label": positive_label,
    }
}
joblib.dump(churn_pipeline, MODEL_DIR / "churn_pipeline.joblib")

# 2. TotalCharges regression model + preprocessing in one Pipeline.
reg_df = df.dropna(subset=REGRESSION_FEATURES + ["TotalCharges"]).copy()
X_reg = reg_df[REGRESSION_FEATURES]
y_reg = reg_df["TotalCharges"].astype(float)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
reg_pipeline = Pipeline(steps=[
    ("preprocessor", make_preprocessor(REGRESSION_FEATURES, [])),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42)),
])
reg_pipeline.fit(Xr_train, yr_train)
reg_pred = reg_pipeline.predict(Xr_test)
metrics["regression"] = {
    "mae": float(mean_absolute_error(yr_test, reg_pred)),
    "rmse": float(np.sqrt(mean_squared_error(yr_test, reg_pred))),
    "r2": float(r2_score(yr_test, reg_pred)),
    "test_size": int(len(yr_test)),
}
joblib.dump(reg_pipeline, MODEL_DIR / "totalcharges_pipeline.joblib")

# 3. Segmentation model: scale the two numeric features, then K-Means.
seg_df = df.dropna(subset=SEGMENTATION_FEATURES).copy()
X_seg = seg_df[SEGMENTATION_FEATURES].astype(float)
segmentation_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42, n_init=10)),
])
segmentation_pipeline.fit(X_seg)
joblib.dump(segmentation_pipeline, MODEL_DIR / "segmentation_pipeline.joblib")

with open(MODEL_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("Training complete.")
print(f"Saved model files in: {MODEL_DIR}")
print("Classification metrics:", metrics["classification"])
print("Regression metrics:", metrics["regression"])
