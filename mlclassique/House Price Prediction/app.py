import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv("house_prices.csv")

print("Dataset:")
print(df)

print("\nDataset Description:")
print(df.describe())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()


# Check numerical and categorical features
num_features = df.select_dtypes(include=np.number).columns
cat_features = df.select_dtypes(include="object").columns

print("\nNumerical features:")
print(num_features)

print("\nCategorical features:")
print(cat_features)


# Check missing values
missing_values = df.isna().sum()

print("\nNumber of missing values:")
print(missing_values)


# Check duplicated rows
duplicated_values = df.duplicated().sum()

print("\nNumber of duplicated rows:")
print(duplicated_values)


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Target variable
print("\nHouse Prices:")
print(df["price"])


# Average price
average_price = df["price"].mean()

print("\nAverage price:")
print(average_price)


# Maximum price
max_price = df["price"].max()

print("\nMaximum price:")
print(max_price)


# Minimum price
min_price = df["price"].min()

print("\nMinimum price:")
print(min_price)


# Correlation matrix
correlation = df.corr(numeric_only=True)

print("\nCorrelation matrix:")
print(correlation)

print("\nCorrelation with price:")
print(correlation["price"])


# ------------------------------------------------------------
# Price Distribution
# ------------------------------------------------------------

plt.hist(df["price"], bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")

plt.show()


# ------------------------------------------------------------
# Area vs Price
# ------------------------------------------------------------

plt.scatter(df["area"], df["price"])

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Area vs House Price")

plt.show()


# ------------------------------------------------------------
# Bedrooms vs Price
# ------------------------------------------------------------

sns.boxplot(x="bedrooms", y="price", data=df)

plt.xlabel("Number of Bedrooms")
plt.ylabel("Price")
plt.title("House Price by Number of Bedrooms")

plt.show()


# ------------------------------------------------------------
# Bathrooms vs Price
# ------------------------------------------------------------

sns.boxplot(x="bathrooms", y="price", data=df)

plt.xlabel("Number of Bathrooms")
plt.ylabel("Price")
plt.title("House Price by Number of Bathrooms")

plt.show()


# ------------------------------------------------------------
# Outliers Detection
# ------------------------------------------------------------

sns.boxplot(data=df[num_features])

plt.title("Outliers Detection")

plt.show()


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================

# Remove duplicated rows
df = df.drop_duplicates()


# Separate features and target
x = df.drop("price", axis=1)
y = df["price"]


# IMPORTANT:
# Recalculate numerical and categorical features
# using X only, because price is the target.

num_features = x.select_dtypes(include=np.number).columns
cat_features = x.select_dtypes(include="object").columns


print("\nNumerical features used for X:")
print(num_features)

print("\nCategorical features used for X:")
print(cat_features)


# ------------------------------------------------------------
# Handle missing numerical values
# ------------------------------------------------------------

x[num_features] = x[num_features].fillna(
    x[num_features].median()
)


# ------------------------------------------------------------
# Handle missing categorical values
# ------------------------------------------------------------

x[cat_features] = x[cat_features].fillna("unknown")


# ------------------------------------------------------------
# Train/Test Split
# ------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)


print("\nX_train shape:")
print(x_train.shape)

print("\nX_test shape:")
print(x_test.shape)

print("\ny_train shape:")
print(y_train.shape)

print("\ny_test shape:")
print(y_test.shape)


# ============================================================
# 4. ENCODING CATEGORICAL FEATURES
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# Learn categories from training data
encoder.fit(x_train[cat_features])


# Transform training categorical features
x_train_encoded = encoder.transform(
    x_train[cat_features]
)


# Transform test categorical features
x_test_encoded = encoder.transform(
    x_test[cat_features]
)


# Convert encoded arrays into DataFrames
x_train_encoded = pd.DataFrame(
    x_train_encoded,
    columns=encoder.get_feature_names_out(cat_features),
    index=x_train.index
)

x_test_encoded = pd.DataFrame(
    x_test_encoded,
    columns=encoder.get_feature_names_out(cat_features),
    index=x_test.index
)


# ============================================================
# 5. COMBINE NUMERICAL + CATEGORICAL FEATURES
# ============================================================

# Numerical features
x_train_num = x_train[num_features]
x_test_num = x_test[num_features]


# Combine numerical and encoded categorical features
x_train_final = pd.concat(
    [x_train_num, x_train_encoded],
    axis=1
)

x_test_final = pd.concat(
    [x_test_num, x_test_encoded],
    axis=1
)


print("\nFinal training features:")
print(x_train_final)

print("\nFinal test features:")
print(x_test_final)


# ============================================================
# 6. MODEL BUILDING - LINEAR REGRESSION
# ============================================================

# Create the model
model = LinearRegression()


# Train the model
model.fit(
    x_train_final,
    y_train
)


# ============================================================
# 7. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    x_test_final
)


print("\nPredicted prices:")
print(y_pred)


# Actual vs predicted
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted:")
print(comparison)
# ============================================================
# Polynomial Regression model
# ============================================================