
# 🛡️ Hate Speech / Toxic Comment Detection

### 🎓 MSc Data Science and Management | IIT Ropar
Course: Data Mining  
**Group Members**:  
- Siddharth Kaushik (2024dss1019)  
- Saif Saleem (2024dss1015)  
- Ujjawal Singh (2024dss1023)  
- Ayush Kumar (2024dss1004)

---

## 📘 Project Description
This project aims to detect hate speech and toxic content using advanced machine learning and deep learning models. The system supports **multi-label classification** for categories:
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`

The detection works for **user-entered text** and **image-based text** via OCR.

---

## 📂 Features

- ✅ Real-time text classification
- ✅ OCR-based screenshot upload and detection using EasyOCR
- ✅ Advanced visualizations:
  - Label correlation heatmap
  - Toxicity vs. comment length
  - Spam ratio vs. toxicity
- ✅ Model performance dashboard
- ✅ Fully Streamlit Cloud compatible

---

## 🧠 Models Compared

| Model                     | Mean AUC ROC |
|--------------------------|--------------|
| Logistic Regression (BR) | 0.73         |
| Logistic Regression (CC) | 0.76         |
| Extra Trees              | 0.93         |
| XGBoost                  | 0.96         |
| LSTM                     | 0.97         |
| BERT                     | 0.985        |

---

## 🧼 Preprocessing

- Text cleaning (punctuation, links, IPs, casing)
- Stop word removal, stemming/lemmatization
- TF-IDF feature extraction

---

## 🔍 Advanced Techniques

- EasyOCR (works with Streamlit Cloud)
- Spam word % estimation
- Screenshot text classification
- Visualization of multi-label imbalance and correlation

---

## 🚀 How to Run

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Launch the app:
```bash
streamlit run app.py
```

---

## 📝 Dataset
We use the Jigsaw Toxic Comment Classification dataset with over 160k labeled comments.

---

## 🏁 Future Improvements
- Integrate Transformer-based models (BERT, RoBERTa)
- Deploy as a browser extension or moderation API
- Add multilingual support and real-time user behavior monitoring

