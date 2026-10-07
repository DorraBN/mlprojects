import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. Understanding the Dataset
# ============================================================

df = pd.read_csv("dataset.csv")

print(df.head())

print(df.info())

print("Shape :", df.shape)

print("Null values :")
print(df.isna().sum())

print("Duplicated rows :", df.duplicated().sum())

numfeatures = df.select_dtypes(include=np.number)
catfeatures = df.select_dtypes(include="object")

print("Numerical features :")
print(numfeatures.columns)

print("Categorical features :")
print(catfeatures.columns)

print("Statistical Summary :")
print(df.describe())


# ============================================================
# 2. Data Preparation
# ============================================================

# Remove CustomerID
features = df.drop("CustomerID", axis=1)

# Standardization
scaler = StandardScaler()

features_scaled = scaler.fit_transform(features)

print("Scaled features :")
print(features_scaled)


# ============================================================
# 3. K-Means Clustering
# ============================================================

# K-Means with k = 2

model = KMeans(
    n_clusters=2,
    random_state=42,
    n_init="auto"
)

labels = model.fit_predict(features_scaled)

print("Labels for k=2 :")
print(labels)


# Test different values of k from 2 to 10

inertias = []

for k in range(2, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init="auto"
    )

    labels = model.fit_predict(features_scaled)

    inertias.append(model.inertia_)

print("Inertias :")
print(inertias)


# Elbow Curve

plt.plot(range(2, 11), inertias, marker="o")

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.show()


# ============================================================
# Final K-Means Model
# ============================================================

# Choose the optimal k from the Elbow Curve


optimal_k = 5

model = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init="auto"
)

labels = model.fit_predict(features_scaled)

print("Final cluster labels :")
print(labels)


# Add cluster labels to original dataset

df["cluster"] = labels

print(df.head())


# ============================================================
# 4. Visualization
# ============================================================

# Annual Income vs Spending Score

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="Annual_Income",
    y="Spending_Score",
    hue="cluster",
    palette="viridis"
)

plt.title("Customer Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")

plt.show()


# ============================================================
# 5. Evaluation
# ============================================================

# Silhouette Score

silhouette = silhouette_score(
    features_scaled,
    labels
)

print("Silhouette Score :", silhouette)