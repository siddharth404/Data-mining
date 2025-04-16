
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🛡️ Hate Speech / Toxic Comment Detection Dashboard")

st.markdown("""
### 📘 Project: Hate Speech / Toxic Comment Detection  
Developed as part of **Data Mining** course by MSc Data Science and Management students at **IIT Ropar**.  
**Group Members:**  
- Siddharth Kaushik (2024dss1019)  
- Saif Saleem (2024dss1015)  
- Ujjawal Singh (2024dss1023)  
- Ayush Kumar (2024dss1004)  
""")

@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Toxicity Threshold", 0.1, 1.0, 0.5, 0.05)
simulate = st.sidebar.button("▶️ Start Real-time Comment Stream")

sample_comments = [
    "I love this community!",
    "You are so stupid and ugly.",
    "Let's have a great day everyone!",
    "You idiot, just shut up.",
    "Have a wonderful weekend 😊",
    "I will kill you if you show up again.",
    "This is nonsense and offensive.",
    "That's a great idea, thanks for sharing!",
    "You're a complete moron.",
    "Blessings to everyone."
]

count_data = {label: 0 for label in labels}
time_series = []

if simulate:
    st.subheader("🟢 Real-Time Comment Stream")
    stream_placeholder = st.empty()
    chart_placeholder = st.empty()

    for _ in range(20):
        comment = random.choice(sample_comments)
        vectorized = vectorizer.transform([comment])
        try:
            probs = model.predict_proba(vectorized)
            predictions = [int(p[1] >= threshold) if len(p) > 1 else 0 for p in probs]
        except Exception as e:
            st.error("Prediction error: " + str(e))
            predictions = [0] * len(labels)

        detected_labels = [labels[i] for i, pred in enumerate(predictions) if pred == 1]
        display_labels = ", ".join(detected_labels) if detected_labels else "Clean"

        for i, pred in enumerate(predictions):
            count_data[labels[i]] += pred
        time_series.append((time.time(), sum(predictions)))

        with stream_placeholder.container():
            st.markdown(f"**Comment:** {comment}")
            st.markdown(f"**Prediction:** `{display_labels}`")
            st.markdown("---")
        time.sleep(1)

    with chart_placeholder.container():
        st.subheader("📊 Toxic Tag Frequency")
        df = pd.DataFrame(list(count_data.items()), columns=["Label", "Count"])
        st.bar_chart(df.set_index("Label"))

st.subheader("🔍 Bias and Fairness Analysis (Simulation)")
identity_keywords = {
    "gender": ["he", "she", "man", "woman"],
    "race": ["black", "white", "asian"],
    "religion": ["muslim", "christian", "jewish"]
}
bias_data = {key: 0 for key in identity_keywords}

for category, keywords in identity_keywords.items():
    for word in keywords:
        sentence = f"This {word} is bad."
        vec = vectorizer.transform([sentence])
        pred = model.predict(vec)[0]
        if any(pred):
            bias_data[category] += 1

bias_df = pd.DataFrame(list(bias_data.items()), columns=["Category", "Flagged Count"])
st.dataframe(bias_df)

st.markdown("---")
st.info("📘 For more info, read the full report in the `README.md`.")
