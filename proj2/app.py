import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


df=pd.read_csv("dataset.csv")
"""
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



churn_percent = df["Churn"].value_counts(normalize=True) * 100

print("Pourcentage de chaque classe:")
print(churn_percent)

plt.pie(
    churn_percent,
    labels=churn_percent.index,
    autopct="%1.1f%%"

)
plt.title("Pourcentage de chaque classe:")
plt.show()





dtenure=df["tenure"]
print("tenure est :", dtenure)
plt.hist(dtenure,bins=30)
plt.title("Distribution de tenure")
plt.xlabel("tenure")
plt.ylabel("Fréquence")
plt.show()


print("la description de monthly charges: ",df["MonthlyCharges"].describe())
plt.hist(df["MonthlyCharges"],bins=30)
plt.title("Analayse de monthlycharges")
plt.xlabel("monthly charges")
plt.ylabel("frequence")
plt.show()


print("la description de TotalCharges: ",df["TotalCharges"].describe())
df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")
print("la description de TotalCharges: ",df["TotalCharges"].describe())
plt.hist(df["TotalCharges"],bins=30)
plt.title("Analayse de TotalCharges")
plt.xlabel("TotalCharges")
plt.ylabel("frequence")
plt.show()


df.boxplot(column="MonthlyCharges",by="Churn")
plt.show()

df.boxplot(column="tenure",by="Churn")
plt.show()


contract_churn=pd.crosstab(df["Contract"],df["Churn"])
print(contract_churn)

contract_churn.plot(kind="bar")

plt.title("Contract selon Churn")
plt.xlabel("Contract")
plt.ylabel("Nombre de clients")
plt.xticks(rotation=0)

plt.show()

"""
#partie3
# 25. TotalCharges -> numérique
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 26. Valeurs manquantes
print(df.isnull().sum())

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)

print("type de TotalCharges:", type(df[["TotalCharges"]]))

# 27. Churn -> 0/1
df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

# 28. X et y
X = df[["TotalCharges"]]
y = df["Churn"]
print("type de TotalCharges:", type(df[["TotalCharges"]]))
print("type de x:", type(X))
print(X)
print(y)
