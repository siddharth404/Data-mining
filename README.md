
# Hate Speech / Toxic Comment Detection Dashboard

## 👨‍🎓 Developed By:
**MSc Data Science and Management Students**  
**Indian Institute of Technology, Ropar**

### 👥 Group Members:
1. Siddharth Kaushik - 2024dss1019  
2. Saif Saleem - 2024dss1015  
3. Ujjawal Singh - 2024dss1023  
4. Ayush Kumar - 2024dss1004  

## 📘 Course:
**Data Mining** — Semester Project

---

## 💡 Project Overview
This project detects hate speech and toxic comments using multiple machine learning and deep learning models. It incorporates:

- Logistic Regression, SVM, Extra Trees, XGBoost
- LSTM and fine-tuned Transformer models like BERT
- Real-time comment stream classification
- OCR-based image upload and analysis
- Exploratory data visualizations
- Model performance dashboard

---

## 📊 Features

- ✅ Upload and classify comments or screenshots
- ✅ Extract text from images using OCR (Tesseract)
- ✅ Custom threshold slider for sensitivity tuning
- ✅ Heatmap of label correlation
- ✅ Box & violin plots of comment length vs toxicity
- ✅ Spam/unique word ratio analysis
- ✅ Side-by-side model AUC ROC comparison
- ✅ Streamlit-ready dashboard for demo or deployment

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Run the App

```bash
streamlit run app.py
```

---

## 📂 Files

- `app.py`: Main dashboard code
- `model.pkl` / `vectorizer.pkl`: Pre-trained classification model
- `requirements.txt`: Python dependencies
- `README.md`: Documentation

---

## 📄 Dataset

Used the Jigsaw Toxic Comment Classification Challenge dataset with labels:
- toxic, severe_toxic, obscene, threat, insult, identity_hate

---

## 🎯 Project Context

This project is a part of the **Data Mining** course for the MSc in Data Science and Management program at **IIT Ropar**, submitted in 2025.

