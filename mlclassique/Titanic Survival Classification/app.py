import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report

# 1. Understanding the Dataset

df = pd.read_csv("titanic.csv")

print(df.head())

print(df.describe())

print("Shape :", df.shape)

print(df.info())

print("Missing values :")
print(df.isna().sum())

print("Duplicated rows :")
print(df.duplicated().sum())
# Identify numerical and categorical features

cat_features = df.select_dtypes(include="object").columns

num_features = df.select_dtypes(include=np.number).columns.drop("Survived")

print("Numerical features :", num_features)
print("Categorical features :", cat_features)


# Handle missing values

df[num_features] = df[num_features].fillna(df[num_features].median())

df[cat_features] = df[cat_features].fillna("unknown")


# Select features and target

X = df.drop("Survived", axis=1)
y = df["Survived"]


# Split the dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Convert categorical features into numerical values

onehotencoder = OneHotEncoder(
    drop="first",
    handle_unknown="ignore",
    sparse_output=False
)

X_train_cat = onehotencoder.fit_transform(X_train[cat_features])

X_test_cat = onehotencoder.transform(X_test[cat_features])


# Convert encoded data into DataFrames

X_train_cat_df = pd.DataFrame(
    X_train_cat,
    columns=onehotencoder.get_feature_names_out(cat_features),
    index=X_train.index
)

X_test_cat_df = pd.DataFrame(
    X_test_cat,
    columns=onehotencoder.get_feature_names_out(cat_features),
    index=X_test.index
)


# Combine numerical and categorical features

X_train_final = pd.concat(
    [X_train[num_features], X_train_cat_df],
    axis=1
)

X_test_final = pd.concat(
    [X_test[num_features], X_test_cat_df],
    axis=1
)


# Logistic Regression

model = LogisticRegression(max_iter=1000)

model.fit(X_train_final, y_train)


# Prediction

y_predict = model.predict(X_test_final)

print("Predicted target :")
print(y_predict)
# K-Nearest Neighbors (KNN) classification model
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train_final)
X_test_scaled=scaler.transform(X_test_final)
model_knn=KNeighborsClassifier(n_neighbors=5)
model_knn.fit(X_train_scaled,y_train)
y_predict_knn=model_knn.predict(X_test_scaled)
print("predicted target by knn :",y_predict_knn)
#Decision Tree classification model
model_dt=DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_final,y_train)
y_predict_dt=model_dt.predict(X_test_final)
print("predicted target by decision tree :",y_predict_dt)
#Random Forest classification model
model_rf=RandomForestClassifier(n_estimators=100,random_state=42)
model_rf.fit(X_train_final,y_train)
y_predict_rf=model_rf.predict(X_test_final)
print("predicted target by random forest :",y_predict_rf)
#Gradient Boosting classification model
model_gb=GradientBoostingClassifier(n_estimators=100,random_state=42)
model_gb.fit(X_train_final,y_train)
y_predict_gb=model_gb.predict(X_test_final)
print("predicted target by gradient boosting :",y_predict_gb)
#4. Model Evaluation
#the Accuracy of each model
accuracy_lr=accuracy_score(y_test,y_predict)
accuracy_knn=accuracy_score(y_test,y_predict_knn)
accuracy_dt=accuracy_score(y_test,y_predict_dt)
accuracy_rf=accuracy_score(y_test,y_predict_rf)
accuracy_gb=accuracy_score(y_test,y_predict_gb)
#the Precision of each model
precision_lr=precision_score(y_test,y_predict)
precision_knn=precision_score(y_test,y_predict_knn)
precision_dt=precision_score(y_test,y_predict_dt)
precision_rf=precision_score(y_test,y_predict_rf)
precision_gb=precision_score(y_test,y_predict_gb)
#the Recall of each model
recall_lr=recall_score(y_test,y_predict)
recall_knn=recall_score(y_test,y_predict_knn)
recall_dt=recall_score(y_test,y_predict_dt)
recall_rf=recall_score(y_test,y_predict_rf)
recall_gb=recall_score(y_test,y_predict_gb)
#the F1-Score of each model
f1_lr=f1_score(y_test,y_predict)
f1_knn=f1_score(y_test,y_predict_knn)
f1_dt=f1_score(y_test,y_predict_dt)
f1_rf=f1_score(y_test,y_predict_rf)
f1_gb=f1_score(y_test,y_predict_gb)
#the Confusion Matrix for each model
cm_lr=confusion_matrix(y_test,y_predict)
cm_knn=confusion_matrix(y_test,y_predict_knn)
cm_dt=confusion_matrix(y_test,y_predict_dt)
cm_rf=confusion_matrix(y_test,y_predict_rf)
cm_gb=confusion_matrix(y_test,y_predict_gb)
#Compare all models
print("\nModel Comparison")

print("Logistic Regression:")
print("Accuracy :", accuracy_lr)
print("Precision:", precision_lr)
print("Recall   :", recall_lr)
print("F1-Score :", f1_lr)

print("\nKNN:")
print("Accuracy :", accuracy_knn)
print("Precision:", precision_knn)
print("Recall   :", recall_knn)
print("F1-Score :", f1_knn)

print("\nDecision Tree:")
print("Accuracy :", accuracy_dt)
print("Precision:", precision_dt)
print("Recall   :", recall_dt)
print("F1-Score :", f1_dt)

print("\nRandom Forest:")
print("Accuracy :", accuracy_rf)
print("Precision:", precision_rf)
print("Recall   :", recall_rf)
print("F1-Score :", f1_rf)

print("\nGradient Boosting:")
print("Accuracy :", accuracy_gb)
print("Precision:", precision_gb)
print("Recall   :", recall_gb)
print("F1-Score :", f1_gb)