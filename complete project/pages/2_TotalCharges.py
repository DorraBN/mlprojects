"""Estimate TotalCharges with the saved regression pipeline."""
import joblib
import pandas as pd
import streamlit as st
from preprocessing import MODEL_DIR

st.set_page_config(page_title="Estimate TotalCharges", layout="wide")
st.title("Estimate TotalCharges")
st.write("Estimate the total charges using tenure and monthly charges.")

with st.form("total_charges_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    monthly = st.number_input("MonthlyCharges", min_value=0.0, value=70.0, step=1.0)
    submitted = st.form_submit_button("Estimate TotalCharges")

if submitted:
    if tenure < 0 or monthly < 0:
        st.error("Values must be non-negative.")
    else:
        try:
            model = joblib.load(MODEL_DIR / "totalcharges_pipeline.joblib")
            customer = pd.DataFrame([{"tenure": tenure, "MonthlyCharges": monthly}])
            estimate = float(model.predict(customer)[0])
            st.metric("Estimated TotalCharges", f"{max(0.0, estimate):.2f}")
            st.caption("This is a model estimate, not an invoice or guaranteed amount.")
        except FileNotFoundError:
            st.error("Model not found. Run `python train_models.py` first.")
        except Exception as exc:
            st.error(f"Could not estimate TotalCharges: {exc}")
