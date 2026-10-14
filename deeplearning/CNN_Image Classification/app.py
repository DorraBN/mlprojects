import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.models import Sequential

# trainning data
df_train=pd.read_csv("fashion-mnist_train.csv")
print(df_train.head())
print(df_train.describe())
print(df_train.info())
print(df_train.shape)
print("missing values:",df_train.isna().sum())
print("duplicated values :",df_train.duplicated().sum())
df_train=df_train.drop_duplicates()
catfeatures=df_train.select_dtypes(include="object").columns
numfeatures=df_train.select_dtypes(include=np.number).columns
print("categ features :",catfeatures)
print("num features :",numfeatures)
#test data
df_test=pd.read_csv("fashion-mnist_test.csv")
print(df_test.head())
print(df_test.describe())
print(df_test.info())
print(df_test.shape)
print("missing values:",df_test.isna().sum())
print("duplicated values :",df_test.duplicated().sum())
df_test=df_test.drop_duplicates()
print("duplicated values :",df_test.duplicated().sum())
catfeatures1=df_test.select_dtypes(include="object").columns
numfeatures1=df_test.select_dtypes(include=np.number).columns
print("categ features 1:",catfeatures)
print("num features 1:",numfeatures)
#Separate the target variable label from the pixel features.
x_train=df_train.drop("label",axis=1)
y_train=df_train["label"]
x_test=df_test.drop("label",axis=1)
y_test=df_test["label"]
#Convert the pixel values from Pandas DataFrames to NumPy arrays.
x_train=x_train.to_numpy()
y_train=y_train.to_numpy()
x_test=x_test.to_numpy()
y_test=y_test.to_numpy()
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#Normalize the pixel values from [0, 255] to [0, 1].
x_train = x_train / 255.0 
x_test = x_test / 255.0
#Reshape the pixel data from:784 → 28 × 28 × 1
x_train = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1) 
print("x_train after reshape:", x_train.shape) 
print("x_test after reshape:", x_test.shape)
#Display some sample images using Matplotlib.
for i in range(5):
   plt.imshow(x_train[i].squeeze(), cmap="gray")
   plt.title(df_train["label"][i])
   plt.show()
#Build a CNN using Keras containing:
model = Sequential()

model.add(Conv2D(
    32,
    (3,3),
    activation="relu",
    input_shape=(28,28,1)
))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dense(10, activation="softmax"))


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=30
)

loss, accuracy = model.evaluate(x_test, y_test)

print("Test accuracy:", accuracy)

y_predict = model.predict(x_test)
print(y_predict)
y_pred_classes = np.argmax(y_predict, axis=1)

print(y_pred_classes)