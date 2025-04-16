
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pytesseract
from PIL import Image

st.set_page_config(layout="wide")
st.title("📊 Hate Speech Detection & Model Dashboard")

st.markdown("""
### 🎓 MSc Data Science and Management | IIT Ropar  
Course: Data Mining | Group Members:  
- Siddharth Kaushik (2024dss1019)  
- Saif Saleem (2024dss1015)  
- Ujjawal Singh (2024dss1023)  
- Ayush Kumar (2024dss1004)  
""")

@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

st.sidebar.header("🛠️ Settings")
threshold = st.sidebar.slider("Toxicity Threshold (only applies to probability-based models)", 0.1, 1.0, 0.5, 0.05)

#####################
# 🔍 Image OCR Input
#####################
st.subheader("📷 Text Extraction from Uploaded Screenshot")

uploaded_image = st.file_uploader("Upload a screenshot containing a comment", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Screenshot", use_column_width=True)

    text = pytesseract.image_to_string(image)
    st.write("🔤 Extracted Text:", text)

    vec = vectorizer.transform([text])
    try:
        preds = model.predict(vec)[0]
        detected = [labels[i] for i, val in enumerate(preds) if val]
        st.success("Prediction: " + (", ".join(detected) if detected else "Clean"))
    except Exception as e:
        st.error("Model Error: " + str(e))

##########################
# 📝 Custom User Input
##########################
st.subheader("📝 Classify Your Own Comment")

user_text = st.text_area("Enter a sentence or comment below:")
if st.button("Classify Text"):
    if user_text.strip():
        vec = vectorizer.transform([user_text])
        try:
            preds = model.predict(vec)[0]
            detected = [labels[i] for i, val in enumerate(preds) if val]
            st.success("Prediction: " + (", ".join(detected) if detected else "Clean"))
        except Exception as e:
            st.error("Model Error: " + str(e))
    else:
        st.warning("Please enter a comment to classify.")

#####################
# 📊 Visualizations
#####################
st.subheader("📊 Advanced Dataset Visualizations")

if st.checkbox("Show Correlation Heatmap"):
    df = pd.read_csv("train.csv")
    df.fillna("", inplace=True)
    correlation = df[labels].corr()
    fig, ax = plt.subplots()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.checkbox("Comment Length vs Toxicity (Box + Violin)"):
    df = pd.read_csv("train.csv")
    df["length"] = df["comment_text"].fillna("").apply(lambda x: len(x.split()))
    melted = df.melt(id_vars=["length"], value_vars=labels)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x="value", y="length", data=melted[melted["value"] == 1], ax=axs[0])
    sns.violinplot(x="value", y="length", data=melted[melted["value"] == 1], ax=axs[1])
    axs[0].set_title("Box Plot: Toxic Lengths")
    axs[1].set_title("Violin Plot: Toxic Lengths")
    st.pyplot(fig)

if st.checkbox("Spam / Unique Word Percentage"):
    df = pd.read_csv("train.csv")
    df["unique_percent"] = df["comment_text"].fillna("").apply(lambda x: len(set(x.split())) / (len(x.split()) + 1))
    df["spam"] = df["unique_percent"] < 0.4
    spam_ratio = df.groupby("spam")[labels].mean().T
    fig, ax = plt.subplots()
    spam_ratio.plot(kind="bar", ax=ax, title="Spam vs Toxicity Rate", figsize=(10, 5))
    st.pyplot(fig)

#####################
# 📊 Model Comparison Dashboard
#####################
st.subheader("🤖 Model Comparison (Simulated AUC Scores)")

# Simulated AUCs as per your report
auc_data = {
    "Model": ["SVM (BR)", "SVM (CC)", "LogReg (BR)", "LogReg (CC)", "Extra Trees", "XGBoost", "LSTM", "BERT"],
    "AUC ROC": [0.66, 0.67, 0.73, 0.76, 0.93, 0.96, 0.97, 0.985]
}
auc_df = pd.DataFrame(auc_data)

fig, ax = plt.subplots()
sns.barplot(data=auc_df, x="Model", y="AUC ROC", palette="mako", ax=ax)
ax.set_ylim(0.6, 1.0)
st.pyplot(fig)
st.dataframe(auc_df.set_index("Model"))

st.markdown("---")
st.info("📘 For more info, read the full report in the `README.md`.")
