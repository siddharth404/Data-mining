
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pytesseract

st.set_page_config(layout="wide")
st.title("🛡️ Hate Speech / Toxic Comment Detection Dashboard")

st.markdown("""
### 🎓 MSc Data Science and Management | IIT Ropar  
Course: Data Mining | Group Members:  
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

######################
# 📝 Custom Input
######################
st.subheader("✍️ Enter Text for Classification")
user_text = st.text_area("Write a comment to classify:")
if st.button("Classify Text"):
    if user_text.strip():
        vec = vectorizer.transform([user_text])
        preds = model.predict(vec)[0]
        detected = [labels[i] for i, val in enumerate(preds) if val]
        st.success("Prediction: " + (", ".join(detected) if detected else "Clean"))
    else:
        st.warning("Please enter a comment.")

######################
# 📷 OCR Image Upload
######################
st.subheader("📷 Upload Screenshot for Text Detection")
uploaded_image = st.file_uploader("Upload image (screenshot of comment)", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    text = pytesseract.image_to_string(image)
    st.write("Extracted Text:", text)
    vec = vectorizer.transform([text])
    preds = model.predict(vec)[0]
    detected = [labels[i] for i, val in enumerate(preds) if val]
    st.success("Prediction: " + (", ".join(detected) if detected else "Clean"))

######################
# 📊 Visualizations
######################
st.subheader("📈 Dataset Visualizations")
if st.checkbox("Show Label Correlation Heatmap"):
    df = pd.read_csv("train.csv").fillna("")
    fig, ax = plt.subplots()
    sns.heatmap(df[labels].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if st.checkbox("Comment Length vs Toxicity"):
    df = pd.read_csv("train.csv").fillna("")
    df["length"] = df["comment_text"].apply(lambda x: len(x.split()))
    melted = df.melt(id_vars=["length"], value_vars=labels)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=melted[melted["value"] == 1], x="variable", y="length", ax=axs[0])
    sns.violinplot(data=melted[melted["value"] == 1], x="variable", y="length", ax=axs[1])
    axs[0].set_title("Box Plot")
    axs[1].set_title("Violin Plot")
    st.pyplot(fig)

if st.checkbox("Spam / Unique Word Analysis"):
    df = pd.read_csv("train.csv").fillna("")
    df["unique_ratio"] = df["comment_text"].apply(lambda x: len(set(x.split())) / (len(x.split()) + 1))
    df["is_spammy"] = df["unique_ratio"] < 0.4
    spam_toxicity = df.groupby("is_spammy")[labels].mean().T
    fig, ax = plt.subplots()
    spam_toxicity.plot(kind="bar", ax=ax)
    st.pyplot(fig)

######################
# 🧪 Model Comparison
######################
st.subheader("🤖 Model Comparison (Simulated AUC)")
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
