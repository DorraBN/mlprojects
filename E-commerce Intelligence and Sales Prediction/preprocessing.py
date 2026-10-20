
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "online_retail_II.xlsx"
MODEL_DIR = BASE_DIR / "models"


def load_dataset(path=DATA_PATH):
    df = pd.read_excel(path)

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["Quantity"] = pd.to_numeric(
        df["Quantity"], errors="coerce"
    )

    df["Price"] = pd.to_numeric(
        df["Price"], errors="coerce"
    )

    return df


def inspect_dataset(df):
    missing_values = df.isnull().sum()
    duplicated_values = df.duplicated().sum()

    return missing_values, duplicated_values


def calculate_revenue(df):
    df = df.copy()

    df["Revenue"] = df["Quantity"] * df["Price"]

    return df


def extract_date(df):
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(
        df["InvoiceDate"], errors="coerce"
    )

    df["month"] = df["InvoiceDate"].dt.month_name()
    df["year"] = df["InvoiceDate"].dt.year
    df["day"] = df["InvoiceDate"].dt.day_name()

    return df
df=load_dataset(DATA_PATH)


def prepare_customers(df):
    df = df.copy()
    df.columns = df.columns.str.strip()

    df = calculate_revenue(df)
    df = df.dropna(subset=["Customer ID"]).copy()

    customers = df.groupby("Customer ID").agg(
        TotalQuantity=("Quantity", "sum"),
        TotalRevenue=("Revenue", "sum"),
        NumberInvoices=("Invoice", "nunique"),
        NumberProducts=("StockCode", "nunique")
    )

    return customers



def prepare_dailysales(df):
    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Revenue"] = df["Quantity"] * df["Price"]

    daily_sales = (
        df.groupby(df["InvoiceDate"].dt.date)["Revenue"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = ["Date", "DailyRevenue"]
    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])

    daily_sales = daily_sales.sort_values("Date")

    daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek
    daily_sales["DayOfMonth"] = daily_sales["Date"].dt.day
    daily_sales["Month"] = daily_sales["Date"].dt.month
    daily_sales["Quarter"] = daily_sales["Date"].dt.quarter
    daily_sales["Year"] = daily_sales["Date"].dt.year
    daily_sales["IsWeekend"] = (
        daily_sales["DayOfWeek"] >= 5
    ).astype(int)

    return daily_sales








