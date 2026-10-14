import re

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Embedding,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values :",df.isna().sum())
print("dupliacted values :",df.duplicated().sum() )
print(df["sentiment"].value_counts())
df["review"]=df["review"].str.lower()
print(df.head())
df["review"]=df["review"].str.replace(r'[^\w\s]','',regex=True)                                  
print(df.head())
df["tokens"]=df["review"].str.split()
print(df.head())
df["nb"]=df["tokens"].str.len()
print(df.head())
encoder=LabelEncoder()
sentiments=encoder.fit_transform(df["sentiment"])
x=df["review"]
y=sentiments
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
tokenizer=Tokenizer(oov_token="<OOV>")

tokenizer.fit_on_texts(x_train)

x_train_seq=tokenizer.texts_to_sequences(x_train)
x_test_seq=tokenizer.texts_to_sequences(x_test)
# 6. Padding
max_len=6
x_train=pad_sequences(x_train_seq,maxlen=max_len,padding="post",truncating="post")
x_test=pad_sequences(x_test_seq,maxlen=max_len,padding="post",truncating="post")
vocab_size=len(tokenizer.word_index)+1

print("Vocabulary size:", vocab_size)
print("Training shape:", x_train.shape)
print("Testing shape:", x_test.shape)



model=Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=16,
    input_length=max_len
))

model.add(LSTM(32))
model.add(Dense(1,activation="sigmoid"))
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.fit(x_train,y_train,epochs=50,batch_size=10)
model.summary()
y_predict=model.predict(x_test)
loss,accuracy=model.evaluate(x_test,y_test)
new="This product is absolutely amazing"
new=new.lower()
new=re.sub(r'[^/w/s]','',new)
new_seq = tokenizer.texts_to_sequences([new])

new_pad = pad_sequences(
    new_seq, maxlen=max_len, padding="post", truncating="post"
)

prediction = model.predict(new_pad)

predicted_label = int(prediction[0][0] >= 0.5)

sentiment = encoder.inverse_transform([predicted_label])[0]

print("New review:", new)
print("Prediction score:", prediction[0][0])
print("Predicted sentiment:", sentiment)