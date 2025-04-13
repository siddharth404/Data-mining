
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("Toxic Comment Classification")

input_text = st.text_area("Enter a comment to classify:")

if st.button("Classify"):
    if input_text.strip() != "":
        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        result = [labels[i] for i, val in enumerate(prediction) if val == 1]
        st.success("Predicted categories: " + (", ".join(result) if result else "Clean"))
    else:
        st.warning("Please enter some text.")
