from preprocessing import(
    DATA_PATH,
    MODEL_DIR,
        load_dataset,
        prepare_customers,
        prepare_dailysales
)
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df=load_dataset(DATA_PATH)
Customers=prepare_customers(df)
FEATURES=Customers[
    ["TotalQuantity", "TotalRevenue",
     "NumberInvoices", "NumberProducts"]
]
scaler=StandardScaler()
Features_scaled=scaler.fit_transform(FEATURES)
model=KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)
Customers["Cluster"]=model.fit_predict(Features_scaled)
print("Cluster ",Customers.head())
""""
plt.scatter(
    Customers["NumberInvoices"],
    Customers["TotalRevenue"],
    c=Customers["Cluster"]
)
plt.xlabel("Number of Invoices")
plt.ylabel("Total Revenue")
plt.title("Customer Segmentation using K-Means")
plt.show()
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model,MODEL_DIR/"kmeans_model.joblib")
joblib.dump(scaler,MODEL_DIR/"scaler.joblib")
"""

#regression sales prediction
daily_sales=prepare_dailysales(df)
FEATURES_reg = daily_sales.drop(
    columns=["DailyRevenue", "Date"]
)
TARGET_reg=daily_sales["DailyRevenue"]
x_train,x_test,y_train,y_test=train_test_split(FEATURES_reg,TARGET_reg,random_state=42,test_size=0.2)
model_reg=LinearRegression()
model_reg.fit(x_train,y_train)
y_predict=model_reg.predict(x_test)
joblib.dump(model_reg,MODEL_DIR/"model_reg.joblib")