import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score,recall_score,confusion_matrix,precision_score
df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values :",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
print(df.describe())
print(df.shape)
numfeatures =df.select_dtypes(include=np.number).columns
catfeatures=df.select_dtypes(include="object").columns

x=df.drop("Passed",axis=1)
y=df["Passed"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)
model=SVC()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print("y predict :", y_predict)
print(" accuracy :", accuracy_score(y_test,y_predict))
print(" recall :", recall_score(y_test,y_predict))
print(" precision :", precision_score(y_test,y_predict))
print(" f1 score :", f1_score(y_test,y_predict))
print(" confusion matrix :", confusion_matrix (y_test,y_predict))