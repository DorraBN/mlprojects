import numpy as np
import pandas as pd
df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values :",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
