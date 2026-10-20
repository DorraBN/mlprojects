import streamlit as st
import joblib
from preprocessing import (MODEL_DIR,load_dataset,prepare_customers)
st.set_page_config(page_title="Customer Segmentation",layout="wide")
st.title("Customer Segmentation")
model=joblib.load(MODEL_DIR/"kmeans_model.joblib")
scaler=joblib.load(MODEL_DIR/"scaler.joblib")
@st.cache_data
def get_customers():
    df=load_dataset()
    customers=prepare_customers(df)
    return customers
Customers=get_customers()
features = Customers[
    [
        "TotalQuantity",
        "TotalRevenue",
        "NumberInvoices",
        "NumberProducts"
    ]
]
features_scaled=scaler.transform(features)
Customers["Cluster"]=model.predict(features_scaled)
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(Customers))
col2.metric("Number of Clusters", Customers["Cluster"].nunique())
col3.metric("Total Revenue", f"{Customers['TotalRevenue'].sum():,.2f}")

st.divider()
st.bar_chart(Customers['TotalRevenue'])
