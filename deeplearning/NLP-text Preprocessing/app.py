import re

import pandas as pd
import numpy as np

from collections import Counter
df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values:",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
dfc=df.copy()
dfc["review"]=dfc["review"].str.lower()
print(dfc.head())
dfc["review"]=dfc["review"].str.replace(r'[^\w\s]','',regex=True)
print(dfc.head())



dfc["tokens"]=dfc["review"].str.split()
print(dfc.head())
stop_words=["the","is","a"]
dfc["tokens"]=dfc["tokens"].apply(
    lambda words: [word for word in words if word not in stop_words]
)
print(dfc.head())


dfc["cleaned_reviews"]=dfc["tokens"].apply(
    lambda words : " ".join(words)
)
print(dfc.head())

print("comparison of reviews")
print(pd.DataFrame({
    "original":df["review"],
    "cleaned":dfc["cleaned_reviews"]
}).head())

dfc["nb"]=dfc["tokens"].str.len()
print(dfc[["cleaned_reviews","nb"]].head())
allwords=[]
for tokens in dfc["tokens"]:
    allwords.extend(tokens)
word_frequency=Counter(allwords)
print("mostr common words ",word_frequency.most_common(10))


# 11. Calculate the frequency of positive and negative sentiments
print(df["sentiment"].value_counts())
new_review='THE PRODUCT IS AMAZING!!!'
new_review=new_review.lower()
new_review=re.sub(r'[^\w\s]','',new_review)
print(new_review)
tokenss=new_review.split()
print(tokenss)
new_tokens=[word for word in tokenss  if word not in stop_words]
print(new_tokens)
print("Cleaned new review:", " ".join(new_tokens))