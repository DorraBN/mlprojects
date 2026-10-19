"""Main Streamlit app: customer dashboard and churn prediction."""
import joblib
import pandas as pd
import streamlit as st

from preprocessing import (
    BASE_DIR, DATA_PATH, MODEL_DIR, CHURN_FEATURES,
    load_dataset, validate_required_columns
)

st.set_page_config(page_title="Customer Churn", page_icon="📊", layout="wide")


@st.cache_data
def get_data():
    return load_dataset()


@st.cache_resource
def get_churn_model():
    return joblib.load(MODEL_DIR / "churn_pipeline.joblib")


st.title("Customer Churn Prediction")
st.write("Dashboard and prediction tools for the Telco customer dataset.")

if not DATA_PATH.exists():
    st.error("dataset.csv was not found. Put it in the same folder as app.py.")
    st.stop()

try:
    df = get_data()
    validate_required_columns(
        df, ["Churn", "tenure", "Contract", "InternetService", "MonthlyCharges", "TotalCharges"]
    )
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

# Dashboard
st.header("1. Customer dashboard")
total_customers = len(df)
churn_rate = (df["Churn"].astype(str).str.strip() == "Yes").mean() * 100
avg_monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce").mean()
avg_tenure = pd.to_numeric(df["tenure"], errors="coerce").mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers", f"{total_customers:,}")
c2.metric("Churn rate", f"{churn_rate:.1f}%")
c3.metric("Average monthly charges", f"{avg_monthly:.2f}" if pd.notna(avg_monthly) else "N/A")
c4.metric("Average tenure (months)", f"{avg_tenure:.1f}" if pd.notna(avg_tenure) else "N/A")

left, right = st.columns(2)
with left:
    st.subheader("Churn distribution")
    churn_counts = df["Churn"].astype(str).value_counts()
    st.bar_chart(churn_counts)
with right:
    st.subheader("Monthly charges distribution")
    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce").dropna()
    if not charges.empty:
        st.bar_chart(charges.value_counts(bins=10, sort=False).sort_index())

st.divider()

# Churn prediction form
st.header("2. Predict churn for a new customer")
st.caption("The model uses four features: tenure, Contract, InternetService, MonthlyCharges.")
with st.form("churn_form"):
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
    contract_options = sorted(df["Contract"].dropna().astype(str).unique().tolist())
    internet_options = sorted(df["InternetService"].dropna().astype(str).unique().tolist())
    contract = st.selectbox("Contract", contract_options)
    internet_service = st.selectbox("InternetService", internet_options)
    monthly_charges = st.number_input(
        "MonthlyCharges", min_value=0.0, max_value=10000.0, value=70.0, step=1.0
    )
    submitted = st.form_submit_button("Predict churn")

if submitted:
    try:
        if tenure < 0 or monthly_charges < 0:
            st.error("Tenure and MonthlyCharges must be non-negative.")
        else:
            customer = pd.DataFrame([{
                "tenure": tenure,
                "Contract": contract,
                "InternetService": internet_service,
                "MonthlyCharges": monthly_charges,
            }], columns=CHURN_FEATURES)
            model = get_churn_model()
            prediction = str(model.predict(customer)[0])
            st.subheader("Prediction result")
            if prediction.lower() == "yes":
                st.warning("The model predicts that this customer may churn.")
            else:
                st.success("The model predicts that this customer is likely to stay.")
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                class_names = model.named_steps["model"].classes_
                probability_table = pd.DataFrame({
                    "Class": class_names,
                    "Probability": probabilities
                })
                st.dataframe(probability_table, hide_index=True, use_container_width=True)
    except FileNotFoundError:
        st.error("Saved models are missing. Run `python train_models.py` first.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.info(
    "Use the sidebar to open TotalCharges, Segmentation, and Evaluation pages. "
    "Predictions are estimates, not guarantees."
)
