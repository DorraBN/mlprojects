import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("Sample-Superstore.csv")

print(df.head())
print(df.info())

# 1. Nombre de commandes
nbcommande = df["Order ID"].nunique()
print("Nombre de commandes :", nbcommande)

# 2. Total des ventes
tot = df["Sales"].sum()
print("Total des ventes :", tot)

# 3. Total du profit
proftot = df["Profit"].sum()
print("Total du profit :", proftot)

# 4. Vente moyenne
print("Valeur moyenne d'une vente :", df["Sales"].mean())

# 5. Profit moyen
print("Valeur moyenne du profit :", df["Profit"].mean())

# 6. Quantité totale vendue
print("Quantité totale vendue :", df["Quantity"].sum())

# 7. Remise moyenne
print("Remise moyenne :", df["Discount"].mean())

# 8. Vente maximale
print("Vente maximale :", df["Sales"].max())

# 9. Profit maximal
print("Profit maximal :", df["Profit"].max())

# 10. Nombre de commandes à perte
print(
    "Nombre de commandes à perte :",
    df.loc[df["Profit"] < 0, "Order ID"].nunique()
)

print(df["Order Date"].head())

df["Order Date"] = pd.to_datetime(df["Order Date"])
print(df["Order Date"].head())
sales_by_month = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
print(sales_by_month)
plt.figure(figsize=(10,5))
plt.plot(
    sales_by_month.index.astype(str),
    sales_by_month.values
)

plt.title("Évolution des ventes par mois")
plt.xlabel("Mois")
plt.ylabel("Ventes")
plt.xticks(rotation=45)

plt.show()
# ventes total par categorie
vpc = df.groupby("Category")["Sales"].sum()

print(vpc)

plt.figure(figsize=(8, 5))

plt.bar(
    vpc.index,
    vpc.values
)

plt.title("Ventes par catégorie")
plt.xlabel("Catégorie")
plt.ylabel("Ventes")


plt.show()
# Profit total par région
profpr=df.groupby("Region")["Profit"].sum()
print(profpr)
plt.figure(figsize=(8,5))
plt.bar(

    profpr.index,
    profpr.values
)
plt.title("Profit par région")
plt.xlabel("Région")
plt.ylabel("Profit")
plt.xticks(rotation=45)
plt.show()


ppv=df.groupby("Product Name")["Sales"].sum().sort_values(ascending=False).head(10)
print("Top 10 produits par ventes :", ppv)
plt.figure(figsize=(10,5))
plt.bar(
    ppv.index,
    ppv.values
)
plt.title("Top 10 produits par ventes")
plt.xlabel("Produit")
plt.ylabel("Ventes")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,5))
plt.scatter(df["Discount"],df["Profit"])
plt.xlabel("remise")
plt.ylabel("profit")

plt.show()





seuilventes=df["Sales"].quantile(0.95)
print("seuil des ventes :", seuilventes)



anormalies=df[df["Sales"]>seuilventes]
sal=anormalies["Sales"]
print("sales",sal)


seuilprofit = df["Profit"].quantile(0.05)

profn = df[df["Profit"] < seuilprofit]

print("Seuil profit :", seuilprofit)
print("Profits extrêmement négatifs :")
print(profn["Profit"])


seuilremis=df["Discount"].quantile(0.95)
remis=df[df["Discount"]>seuilremis]
print("remise",remis["Discount"])
