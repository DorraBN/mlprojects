"""Customer segmentation using the saved K-Means pipeline."""
import joblib
import pandas as pd
import streamlit as st
from preprocessing import DATA_PATH, MODEL_DIR, load_dataset

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation")
st.write("Group customers using tenure and MonthlyCharges.")

if not DATA_PATH.exists():
    st.error("dataset.csv was not found.")
    st.stop()

try:
    df = load_dataset()
    features = df[["tenure", "MonthlyCharges"]].apply(pd.to_numeric, errors="coerce")
    valid = features.dropna()
    model = joblib.load(MODEL_DIR / "segmentation_pipeline.joblib")
    labels = model.predict(valid)
    result = valid.copy()
    result["Cluster"] = labels.astype(str)
    st.subheader("Customers by cluster")
    st.bar_chart(result["Cluster"].value_counts().sort_index())
    st.subheader("Cluster visualization")
    st.scatter_chart(result, x="tenure", y="MonthlyCharges", color="Cluster")
    st.dataframe(result.head(100), use_container_width=True)
except FileNotFoundError:
    st.error("Segmentation model missing. Run `python train_models.py` first.")
except Exception as exc:
    st.error(f"Segmentation page failed: {exc}")
