import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments,Trainer

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import re
from transformers import AutoTokenizer
df=pd.read_csv("dataset.csv")
print(df.head())
print("missing values :",df.isna().sum())
print("duplicated values :",df.duplicated().sum())
print("frequency of categories :",df["category"].value_counts())
print("shape:",df.shape)


#Part 2 — Text preprocessing
df["text"]=df["text"].str.lower()
df["text"]=df["text"].str.replace(r'[^\w\s]','',regex=True)
df["tokens"]=df["text"].str.split()
print(df.head())
x=df["text"]
y=df["category"]
encoder=LabelEncoder()
yencoded=encoder.fit_transform(y)
print("target encoded :",yencoded)
print(encoder.classes_)

x_train, x_test, y_train, y_test = train_test_split(
    x, yencoded,
    test_size=0.2,
    random_state=42,
    stratify=yencoded
)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(
    x_train.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

test_encodings = tokenizer(
    x_test.tolist(),
    truncation=True,
    padding=True,
    max_length=128
)

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(value[idx])
            for key, value in self.encodings.items()
        }

        item["labels"] = torch.tensor(
            int(self.labels[idx]),
            dtype=torch.long
        )

        return item

train_dataset = TextDataset(train_encodings, y_train)
test_dataset = TextDataset(test_encodings, y_test)

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(encoder.classes_)
)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

results = trainer.evaluate()
print(results)
