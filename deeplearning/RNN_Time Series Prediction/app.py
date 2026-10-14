import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values:",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
print(df.describe())
x=df["Day"]
y=df["Sales"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scale=MinMaxScaler()
y_train=scale.fit_transform(y_train)
y_test=scale.transform(y_test)
model=Sequential()
model.add(Dense(32,activation="relu",input_shape=(x_train.shape[1],)))
model.add(Dense(8,activation="relu"))
model.add(Dense(1,activation="linear"))
model.compile(
    optimize="adam",
    loss="mean_squared_error",
    metrics=["mse"]
)
model.fit(x_train,y_train,epochs=50,batch_size=5)
y_predict=model.predict(x_test)