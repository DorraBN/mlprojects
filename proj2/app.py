import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("dataset.csv")

print(df.head(5))
print(df.tail(5))
print(df.shape)
df.info()
print(df.describe())
dm=df["Churn"]
print("Churn est:", dm)
vm=df.isnull().sum()
print("vm:", vm)
numerical_columns = df.select_dtypes(include=[np.number]).columns
print("Columns numériques:", numerical_columns)
categorical_columns = df.select_dtypes(include=[object]).columns
print('Columns catégoriques:', categorical_columns)
print('nb duplicated :',df.duplicated().sum())


nbyes=df["Churn"].value_counts()["Yes"]
print("nombre de yes:", nbyes)
nbnos=df["Churn"].value_counts()["No"]
print("nombre de no:", nbnos)

dtenure=df["tenure"]
print("tenure est :", dtenure)
plt.hist(dtenure,bins=30)
plt.title("Distribution de tenure")
plt.xlabel("tenure")
plt.ylabel("Fréquence")
plt.show()


print("Mcharge est :", df["MonthlyCharges"])
plt.hist(df["MonthlyCharges"], bins=30)
plt.title("Distribution des charges mensuelles")
plt.xlabel("Charges mensuelles")
plt.ylabel("Fréquence")
plt.show()
plt.figure(figsize=(10,5))
plt.pie(df["MonthlyCharges"], labels=df["Churn"], autopct='%1.1f%%')
plt.title("Relation entre Churn et MonthlyCharges")
plt.ylabel("Churn")
plt.xlabel("MonthlyCharges")
plt.show()