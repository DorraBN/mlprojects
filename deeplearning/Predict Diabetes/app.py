import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
df=pd.read_csv("dataset.csv")
print(df.head())
print(df.describe())
print("missing values:",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
catfeatures=df.select_dtypes(include="object").columns
numfeatures=df.select_dtypes(include=np.number).columns
print("categ features :",catfeatures)
print("num features :",numfeatures)
x=df.drop("Outcome",axis=1)
y=df["Outcome"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scale=StandardScaler()
x_trainscaled=scale.fit_transform(x_train)
x_testscaled=scale.transform(x_test)
model=Sequential()
model.add(Dense(12,activation="relu",input_shape=(x_trainscaled.shape[1],)))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile( optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"] )
model.fit(x_trainscaled,y_train,epochs=100,batch_size=30)
y_predict=model.predict(x_testscaled)
y_predict=(y_predict>=0.5).astype(int).flatten()
print("y predicted :",y_predict)
print(" accurancy :",accuracy_score(y_test,y_predict) )
new_patient=np.array([[
 2
,125
,70
,30
,110
,32
,30
]])
new_patientscaled=scale.transform(new_patient)
result=(model.predict(new_patientscaled)>=0.5).astype(int).flatten()
print("new_patient result :",result)