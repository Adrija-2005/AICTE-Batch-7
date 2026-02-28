import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load datasets
fake = pd.read_csv("data/Fake.csv", encoding="latin-1")
real = pd.read_csv("data/True.csv", encoding="latin-1")

fake["label"] = 0
real["label"] = 1

data = pd.concat([fake, real])
data = data.sample(frac=1, random_state=42)

X = data["text"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train_vec, y_train)

# Evaluate
print(classification_report(y_test, model.predict(X_test_vec)))

# Save model
joblib.dump(model, "model.jb")
joblib.dump(vectorizer, "vectorizer.jb")

print("Model training complete!")