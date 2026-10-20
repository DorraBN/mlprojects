
import streamlit as st

from preprocessing import (
    DATA_PATH,
    MODEL_DIR,
    load_dataset,
    inspect_dataset,
    calculate_revenue,
    extract_date
)

st.set_page_config(
    page_title="E-commerce Intelligence",
    layout="wide"
)

st.title("E-commerce Intelligence and Sales Prediction")


@st.cache_data
def get_data():
    return load_dataset(DATA_PATH)


try:
    df = get_data()

    # Dataset overview
    st.subheader("1. Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Transactions", len(df))

    with col2:
        st.metric("Total Columns", len(df.columns))

    with col3:
        st.metric("Duplicate Rows", int(df.duplicated().sum()))

    st.dataframe(df, use_container_width=True)

    # Dataset inspection
    st.subheader("2. Data Quality")

    missing, duplicated = inspect_dataset(df)

    missing_df = missing.reset_index()
    missing_df.columns = ["Column", "Missing Values"]

    st.write("Missing values by column")
    st.dataframe(missing_df, use_container_width=True)

    st.write("Total duplicated rows:", int(duplicated))

    # Revenue calculation
    st.subheader("3. Revenue Analysis")

    df_revenue = calculate_revenue(df)

    total_revenue = df_revenue["Revenue"].sum()

    st.metric(
        "Total Revenue",
        f"{total_revenue:,.2f}"
    )

    st.dataframe(df_revenue, use_container_width=True)

    # Date extraction
    st.subheader("4. Date Analysis")

    df_dates = extract_date(df_revenue)

    st.dataframe(
        df_dates[
            ["InvoiceDate", "month", "year", "day"]
        ],
        use_container_width=True
    )

    # Monthly revenue
    st.subheader("5. Monthly Revenue")

    monthly_revenue = (
        df_dates.dropna(subset=["InvoiceDate"])
        .groupby("month", sort=False)["Revenue"]
        .sum()
    )

    st.bar_chart(monthly_revenue)

except FileNotFoundError:
    st.error(
        f"Dataset not found: {DATA_PATH}"
    )

except Exception as e:
    st.error(f"Error loading or processing dataset: {e}")

customer_segmentation=st.Page(
    "pages/CustomerSegmentation.py"
)
sales_prediction=st.Page(
    "pages/SalesPrediction.py"
)