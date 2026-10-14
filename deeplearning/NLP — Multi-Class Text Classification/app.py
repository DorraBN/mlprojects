import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,LSTM
import re
df=pd.read_csv("dataet.csv")
print(df.head())
print("missing values :",df.isna().sum())
nb=df["category"].value_counts()
print("frequency of each category ;",nb)
#Part 2 — Text preprocessing
df["text"]=df["text"].str.lower()
df["text"]=df["text"].str.replace(r'[^\w\s]','',regex=True)

print(df.head())
df["tokens"]=df["text"].str.split()

print(df.head())
df["nb"]=df["tokens"].str.len()
print(df.head())
x=df["text"]
y=df["category"]
encoder=LabelEncoder()
y=encoder.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
tokenizer=Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(x_train)
x_train_seq=tokenizer.texts_to_sequences(x_train)
x_test_seq=tokenizer.texts_to_sequences(x_test)
maxlenth=7
x_train_pad=pad_sequences(x_train_seq,maxlen=maxlenth,padding="post",truncating="post")
x_test_pad=pad_sequences(x_test_seq,maxlen=maxlenth,padding="post",truncating="post")
vocabsize=len(tokenizer.word_index)+1
print("Vocabulary size:", vocabsize)
model=Sequential()
model.add(Embedding(input_dim=vocabsize,output_dim=16,input_length=maxlenth))
model.add(LSTM(32))
model.add(Dense(3,activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train_pad,y_train,epochs=50,batch_size=10)
model.summary()
y_predict = model.predict(x_test_pad)

ypred = y_predict.argmax(axis=1)

ypred_category = encoder.inverse_transform(ypred)

print(ypred_category)
new_text="The football team won the championshipé "
new_text=new_text.lower()
new_text=re.sub(r'[^\w\s]','',new_text)
new_textseq=tokenizer.texts_to_sequences([new_text])
new_textpad=pad_sequences(new_textseq,maxlen=maxlenth,padding="post",truncating="post")
newpredict=model.predict(new_textpad)
newpredict=newpredict.argmax(axis=1)
predict=encoder.inverse_transform(newpredict)[0]
print(predict)