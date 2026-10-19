import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
df = pd.read_csv("dataset.csv")
print(df.head())
print("missing values :", df.isna().sum())
print("duplicated values :", df.duplicated().sum())
print("shape :", df.shape)
df.info()
print(df.describe())

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna(subset=["TotalCharges"])

numf = df.select_dtypes(include=np.number).columns.tolist()
print("number of numerical features is :", numf)

catf = df.select_dtypes(include="object").columns.tolist()
print("number of categorical features is :", catf)

print("descriptive statistics for numerical columns :", df[numf].describe())

df_copy = df.drop("customerID", axis=1)
print("df copy :", df_copy.head())

# 4. Separate features and target
x = df_copy.drop("Churn", axis=1)
y = df_copy["Churn"]

# 5. Encode target variable
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
print("classes :", labelencoder.classes_)

# 6. Identify feature types
num_features = x.select_dtypes(include=np.number).columns.tolist()
cat_features = x.select_dtypes(include="object").columns.tolist()

# 7. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Keep the original DataFrames for leakage-safe cross-validation in Part 4
x_train_raw = x_train.copy()
x_test_raw = x_test.copy()
y_train_raw = y_train.copy()
y_test_raw = y_test.copy()

# 8. Preprocess numerical and categorical features
encoder = OneHotEncoder(handle_unknown="ignore")
x_train_cat = encoder.fit_transform(x_train[cat_features])
x_test_cat = encoder.transform(x_test[cat_features])

scaler = StandardScaler()
x_train_num = scaler.fit_transform(x_train[num_features])
x_test_num = scaler.transform(x_test[num_features])

x_train = hstack([x_train_num, x_train_cat]).tocsr()
x_test = hstack([x_test_num, x_test_cat]).tocsr()

print("shape of x_train :", x_train.shape)
print("x_train :", x_train)
print("shape of y_train :", y_train.shape)
print("y_train :", y_train)

# Part 2 — Exploratory Data Analysis (EDA)
for column in numf:
    plt.hist(df[column])
    plt.title("distribution of " + column)
    plt.show()

plt.hist(df["Churn"])
plt.title("churn distribution")
plt.show()

plt.scatter(df["tenure"], df["TotalCharges"])
plt.xlabel("tenure")
plt.ylabel("TotalCharges")
plt.title("relationship between tenure and TotalCharges")
plt.show()

plt.scatter(df["MonthlyCharges"], df["TotalCharges"])
plt.xlabel("MonthlyCharges")
plt.ylabel("TotalCharges")
plt.title("relationship between MonthlyCharges and TotalCharges")
plt.show()

df.boxplot(column="MonthlyCharges", by="Churn")
plt.show()

# Part 3 — Classical Machine Learning

# Exercise 1: Linear Regression
X_reg = df[["tenure", "MonthlyCharges"]]
y_reg = df["TotalCharges"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

y_pred = reg_model.predict(X_test_reg)
print("y predict :", y_pred)
print("MAE:", mean_absolute_error(y_test_reg, y_pred))
print("MSE:", mean_squared_error(y_test_reg, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred)))
print("R2:", r2_score(y_test_reg, y_pred))

# Actual vs Predicted
plt.figure()
plt.scatter(y_test_reg, y_pred, alpha=0.5)
plt.xlabel("Actual TotalCharges")
plt.ylabel("Predicted TotalCharges")
plt.title("Actual vs Predicted TotalCharges")
plt.show()

# Exercise 2: Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_pol = poly.fit_transform(X_train_reg)
X_test_pol = poly.transform(X_test_reg)

reg_model = LinearRegression()
reg_model.fit(X_train_pol, y_train_reg)

y_pred = reg_model.predict(X_test_pol)
print("y predict :", y_pred)
print("MAE:", mean_absolute_error(y_test_reg, y_pred))
print("MSE:", mean_squared_error(y_test_reg, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred)))
print("R2:", r2_score(y_test_reg, y_pred))

# Exercise 3: Decision Tree Regression
for depth in [2, 3, 5, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    model.fit(X_train_reg, y_train_reg)

    y_train_pred = model.predict(X_train_reg)
    y_test_pred = model.predict(X_test_reg)

    print("\nmax_depth:", depth)
    print("Train R2:", r2_score(y_train_reg, y_train_pred))
    print("Test R2:", r2_score(y_test_reg, y_test_pred))

# Exercise 4: Random Forest Regression
model = RandomForestRegressor(
    n_estimators=100, max_depth=5, random_state=42
)
model.fit(X_train_reg, y_train_reg)

y_predict = model.predict(X_test_reg)

print("y predict:", y_predict)
print("MAE:", mean_absolute_error(y_test_reg, y_predict))
print("MSE:", mean_squared_error(y_test_reg, y_predict))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_predict)))
print("R2:", r2_score(y_test_reg, y_predict))

# Exercise 5: Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("y pred :", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Exercise 6: K-Nearest Neighbors (KNN)
for i in range(1, 9):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("n_neighbors:", i)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1:", f1_score(y_test, y_pred, zero_division=0))

# Exercise 7: Decision Tree Classification
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("y pred :", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Exercise 8: Random Forest Classification
model = RandomForestClassifier(
    n_estimators=100, max_depth=5, random_state=42
)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("y pred :", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Exercise 9: Gradient Boosting
model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
)
model.fit(x_train.toarray(), y_train)
y_pred = model.predict(x_test.toarray())

print("y pred :", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Exercise 10: Gaussian Naive Bayes
model = GaussianNB()
model.fit(x_train.toarray(), y_train)
y_pred = model.predict(x_test.toarray())

print("y pred :", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Exercise 11: Support Vector Machine (SVM)
model = SVC()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("y pred :", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1:", f1_score(y_test, y_pred, zero_division=0))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# Exercise 12: Customer Segmentation with K-Means
X_cluster = df[["tenure", "MonthlyCharges"]]

scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

for i in range(2, 6):
    model = KMeans(n_clusters=i, random_state=42, n_init="auto")
    labels = model.fit_predict(X_cluster_scaled)

    silhouette = silhouette_score(X_cluster_scaled, labels)

    print("Number of clusters:", i)
    print("Silhouette Score:", silhouette)

model = KMeans(n_clusters=3, random_state=42, n_init="auto")
labels = model.fit_predict(X_cluster_scaled)

plt.scatter(
    X_cluster["tenure"],
    X_cluster["MonthlyCharges"],
    c=labels
)
plt.xlabel("tenure")
plt.ylabel("MonthlyCharges")
plt.title("Customer Segmentation with K-Means")
plt.show()

# Exercise 13: Hierarchical Clustering
X_sample = X_cluster_scaled[:200]

model = AgglomerativeClustering(n_clusters=5)
labels = model.fit_predict(X_sample)
print("labels predicted :", labels)

linkage_matrix = linkage(X_sample, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

# Part 4 — Model Optimization and Evaluation
# Use raw data in a Pipeline so preprocessing is fitted separately in each CV fold.
# This helps prevent data leakage.

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_features),
    ("cat", categorical_transformer, cat_features)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 1. Cross-validation: compare classification models with the same F1 metric
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM": SVC()
}

cv_results = []
for name, classifier in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])
    scores = cross_val_score(
        pipe, x_train_raw, y_train_raw, cv=cv, scoring="f1", n_jobs=-1
    )
    cv_results.append((name, scores.mean(), scores.std()))
    print(f"{name} — CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_results.sort(key=lambda item: item[1], reverse=True)
print("\\nModels ranked by mean cross-validation F1:")
for name, mean_score, std_score in cv_results:
    print(f"{name}: mean F1={mean_score:.4f}, std={std_score:.4f}")

# 2. GridSearchCV: test every specified combination
knn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", KNeighborsClassifier())
])

param_grid = {
    "model__n_neighbors": [3, 5, 7, 9, 11],
    "model__weights": ["uniform", "distance"]
}

grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="f1",
    n_jobs=-1
)
grid.fit(x_train_raw, y_train_raw)

print("\\nGridSearchCV best parameters:", grid.best_params_)
print("GridSearchCV best CV F1:", grid.best_score_)
grid_pred = grid.best_estimator_.predict(x_test_raw)
print("GridSearchCV test F1:", f1_score(y_test_raw, grid_pred, zero_division=0))

# 3. RandomizedSearchCV: sample a subset of combinations
random_search = RandomizedSearchCV(
    estimator=knn_pipeline,
    param_distributions={
        "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2]
    },
    n_iter=10,
    cv=cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)
random_search.fit(x_train_raw, y_train_raw)

print("\\nRandomizedSearchCV best parameters:", random_search.best_params_)
print("RandomizedSearchCV best CV F1:", random_search.best_score_)
random_pred = random_search.best_estimator_.predict(x_test_raw)
print("RandomizedSearchCV test F1:", f1_score(y_test_raw, random_pred, zero_division=0))

# Compare the tuned models on the same untouched test set
print("\\nFinal comparison on the test set:")
for name, fitted_model in [
    ("GridSearchCV KNN", grid.best_estimator_),
    ("RandomizedSearchCV KNN", random_search.best_estimator_)
]:
    pred = fitted_model.predict(x_test_raw)
    print(
        f"{name}: "
        f"Accuracy={accuracy_score(y_test_raw, pred):.4f}, "
        f"Precision={precision_score(y_test_raw, pred, zero_division=0):.4f}, "
        f"Recall={recall_score(y_test_raw, pred, zero_division=0):.4f}, "
        f"F1={f1_score(y_test_raw, pred, zero_division=0):.4f}"
    )

# Parameters vs hyperparameters:
# Parameters are learned from data (e.g. model coefficients).
# Hyperparameters are chosen before/during training (e.g. n_neighbors, max_depth).
#
# Overfitting: model performs very well on training data but poorly on unseen data.
# Underfitting: model is too simple and performs poorly on both training and test data.
#
# Data leakage: information from the test set or validation fold influences training.
# Keeping preprocessing inside the Pipeline helps prevent preprocessing leakage.

# Part 5 — Deep Learning with TensorFlow/Keras
# Use numeric target labels and split before fitting the scaler.
features = df[["tenure", "MonthlyCharges"]].copy()
target = LabelEncoder().fit_transform(df["Churn"])

x_train_dl, x_test_dl, y_train_dl, y_test_dl = train_test_split(
    features, target, test_size=0.2, random_state=42, stratify=target
)

scale = StandardScaler()
x_train_scaled = scale.fit_transform(x_train_dl)
x_test_scaled = scale.transform(x_test_dl)

model = Sequential()
model.add(Dense(32, activation="relu", input_shape=(x_train_scaled.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    x_train_scaled,
    y_train_dl,
    validation_split=0.2,
    epochs=10,
    batch_size=30,
    verbose=1
)

# Predict on the test set (not the training set)
y_prob = model.predict(x_test_scaled, verbose=0)
y_predict = (y_prob >= 0.5).astype(int).flatten()

print("\\nDeep Learning results:")
print("Accuracy:", accuracy_score(y_test_dl, y_predict))
print("Precision:", precision_score(y_test_dl, y_predict, zero_division=0))
print("Recall:", recall_score(y_test_dl, y_predict, zero_division=0))
print("F1:", f1_score(y_test_dl, y_predict, zero_division=0))
print("Confusion matrix:\\n", confusion_matrix(y_test_dl, y_predict))
