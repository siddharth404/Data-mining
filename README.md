
# Hate Speech / Toxic Comment Detection

### Group Members:
1. Siddharth Kaushik - 2024dss1019
2. Saif Saleem - 2024dss1015
3. Ujjawal Singh - 2024dss1023
4. Ayush Kumar - 2024dss1004

### MSc Data Science and Management  
Indian Institute of Technology, Ropar  
Course: Data Mining

## 🔍 Project Summary
This project addresses the challenge of detecting toxic comments on social platforms using classical ML, ensemble tree methods, and deep learning (LSTM) with pretrained embeddings. It implements:

- Binary Relevance & Classifier Chains
- Logistic Regression, SVM, Extra Trees, XGBoost
- LSTM with Word2Vec, GloVe, FastText
- Real-time simulation
- Customizable toxicity threshold
- Bias and fairness evaluation

## 📊 Dataset
- 160k+ comments labeled with 6 tags:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate

## 🧪 Features
- Real-time comment stream classification
- Adjustable threshold for toxicity
- Bias testing on identity-sensitive phrases
- Visualization of toxic label distribution

## 🚀 Deployment
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🎓 Academic Context
Submitted for the Data Mining Course, MSc Data Science and Management, IIT Ropar.
