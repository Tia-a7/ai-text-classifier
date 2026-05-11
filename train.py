import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

data = pd.read_csv("dataset.csv")

data["text"] = data["text"].str.lower()

X = data["text"]
y = data["label"]

model = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model berhasil dilatih dan disimpan!")