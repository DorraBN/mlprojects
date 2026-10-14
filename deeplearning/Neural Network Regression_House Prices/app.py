
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


# Part 1 — Data Exploration

df = pd.read_csv("house_prices.csv")

print(df.head())
print(df.describe())

print("Missing values:")
print(df.isna().sum())

print("Duplicated rows:", df.duplicated().sum())
print("Dataset shape:", df.shape)
df.info()


# Part 2 — Data Preprocessing

df_t = df.drop("HouseID", axis=1)

x = df_t.drop("Price", axis=1)
y = df_t["Price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

scale = StandardScaler()

x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

model = Sequential()

model.add(Dense(64, activation="relu",
                input_shape=(x_train.shape[1],)))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

history = model.fit(
    x_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2
)


# Part 4 — Evaluate the Model

y_predict = model.predict(x_test).flatten()

mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_predict)

print("\nModel Evaluation:")
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
# Plot Training Loss and Validation Loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.show()


# Compare Actual and Predicted Prices

plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_predict)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()

comp = pd.DataFrame({
    "Actual Prices": y_test.to_numpy(),
    "Predicted Prices": y_predict
})

print("\nComparison:")
print(comp.head())
# Part 5 — Predict the Price of a New House
newhouse = np.array([[2000, 3, 2, 5]])

newhouse = scale.transform(newhouse)

newhousep = model.predict(newhouse)

print("Predicted price:", newhousep)
