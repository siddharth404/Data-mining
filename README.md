
# Hate Speech / Toxic Comment Detection

### Submitted by:
- Harshit Agrawal (18074019)
- Ashish Kumar (18075068)
- Sachin Srivastava (18075070)

### Under the Guidance of:
**Dr. Bhaskar Biswas**  
Dept. of Computer Science and Engineering  
IIT (BHU) Varanasi

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
Submitted as part of the Data Mining (CSE-362) course, Semester V, 2020.
