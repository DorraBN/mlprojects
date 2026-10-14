import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score,confusion_matrix
df=pd.read_csv("iris.csv")
print(df.head())
print("missing values :",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
print(df.describe())
df=df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)
x=df.drop("species",axis=1)
y=df["species"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
scale=StandardScaler()
x_train=scale.fit_transform(x_train)
x_test=scale.transform(x_test)
encoder=LabelEncoder()
y_train=encoder.fit_transform(y_train)
y_test=encoder.transform(y_test)


model=Sequential()
model.add(Dense(16,activation="relu",input_shape=(x_train.shape[1],)))
model.add(Dense(8,activation="relu"))
model.add(Dense(3,activation="softmax"))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
   
)

model.fit(x_train,y_train,epochs=50,batch_size=30)
loss, accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", accuracy)
y_predictp=model.predict(x_test)
y_predict=np.argmax(y_predictp,axis=1)
print("y predict:",y_predict)
print("accuracy :",accuracy_score(y_test,y_predict))
print("confusion matrix :",confusion_matrix(y_test,y_predict))
newiris = np.array([[5.1, 3.5, 1.4, 0.2]])

newiris = scale.transform(newiris)

prediction_prob = model.predict(newiris)
prediction = np.argmax(prediction_prob, axis=1)

print("Predicted species:", encoder.inverse_transform(prediction)[0])