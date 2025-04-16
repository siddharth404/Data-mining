import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib

# Load dataset
df = pd.read_csv("train.csv")
df.fillna("", inplace=True)

X = df["comment_text"]
y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

# Vectorize
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train model
clf = MultiOutputClassifier(LogisticRegression(max_iter=200))
clf.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(clf, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model training complete. Files saved: model.pkl, vectorizer.pkl")
