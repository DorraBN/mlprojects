
import streamlit as st
import joblib

import sys
from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from preprocessing import (
    DATA_PATH,
    MODEL_DIR,
    load_dataset,
    prepare_dailysales
)
st.set_page_config(page_title="Sales Prediction", layout="wide")
st.title("Sales Prediction")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_DIR / "model_reg.joblib")


@st.cache_data
def get_sales_data():
    df = load_dataset(DATA_PATH)
    return prepare_dailysales(df)


model = load_model()
daily_sales = get_sales_data()

st.subheader("Actual vs Predicted Revenue")

# Predict only once per cached dataset/model version
@st.cache_data
def predict_sales(features):
    return model.predict(features)


features = daily_sales.drop(
    columns=["DailyRevenue", "Date"]
)

daily_sales["predictions"] = predict_sales(features)
col1,col2=st.columns(2)
col1.metric("Total Actual Revenue",daily_sales["DailyRevenue"].sum())
col2.metric("Total predicted Revenue",daily_sales["predictions"].sum())
st.line_chart(
    daily_sales.set_index("Date")[
        ["DailyRevenue","predictions"]
    ]
)

y_true = daily_sales["DailyRevenue"]
y_pred = daily_sales["predictions"]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{mae:,.2f}")
col2.metric("RMSE", f"{rmse:,.2f}")
col3.metric("R² Score", f"{r2:.3f}")
