import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,f1_score,precision_score,confusion_matrix,recall_score
df=pd.read_csv("dataset.csv")
print(df.head())
print("missed values :", df.isna().sum())
print("duplicated values :",df.duplicated().sum())
print(df.shape)
numfeatures=df.select_dtypes(include=np.number).columns
catfeatures=df.select_dtypes(include="object").columns
print(" nilerical features :",numfeatures)
print("categ features :",catfeatures)
print(df.describe() )
x=df.drop("Spam",axis=1)
y=df["Spam"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
print("y predict :", y_predict)
print(" accuracy :", accuracy_score(y_test,y_predict))
print(" recall :", recall_score(y_test,y_predict))
print(" precision :", precision_score(y_test,y_predict))
print(" f1 score :", f1_score(y_test,y_predict))
print(" confusion matrix :", confusion_matrix (y_test,y_predict))