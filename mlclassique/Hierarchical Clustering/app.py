import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values :",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
print(df.describe())
print(df.shape)
numfeatures =df.select_dtypes(include=np.number).columns
catfeatures=df.select_dtypes(include="object").columns
df=df.drop("CustomerID",axis=1)
features=df
scaled=StandardScaler()
df_scaled=scaled.fit_transform(df)
model = AgglomerativeClustering(n_clusters=5)

labels = model.fit_predict(df_scaled)

print("Labels :", labels)

print("Silhouette Score :", silhouette_score(df_scaled, labels))

df["cluster"] = labels


# Scatter Plot

plt.scatter(
    df["Annual_Income"],
    df["Spending_Score"],
    c=labels
)

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Clusters")

plt.show()


# Dendrogram

linkage_matrix = linkage(df_scaled, method="ward")

dendrogram(linkage_matrix)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")

plt.show()