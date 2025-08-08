# 📩 SMS Spam Classifier

A machine learning-based web application that classifies SMS messages as **Spam** or **Not Spam**. Built using Python, Scikit-learn, and Streamlit, this app demonstrates NLP preprocessing, model building, and deployment using a simple and intuitive interface.

---

## 🚀 Demo

🖥️ Live Demo : https://render-esc.onrender.com  
📂 GitHub Repo: https://github.com/devmurarijay13/sms-spam-classifier

---

## 📌 Project Overview

This project performs spam detection on SMS messages using natural language processing (NLP) techniques and a machine learning classifier. The trained model is deployed using **Streamlit** for an interactive frontend.

---

## 🧠 Features

- Binary classification: Spam vs Not Spam
- Text preprocessing with NLTK
- Feature extraction using TF-IDF Vectorizer
- ML model training with Scikit-learn
- Deployment-ready Streamlit web app
- Clean code structure with `app.py`, `model.pkl`, and `vectorizer.pkl`

---

## 🛠️ Tech Stack

| Category          | Tools & Libraries                          |
|-------------------|---------------------------------------------|
| Language          | Python                                      |
| Libraries         | NLTK, Scikit-learn, Streamlit               |
| ML Techniques     | TF-IDF Vectorization, Naive Bayes (or SVM) |
| Dataset           | `sms_spam.csv` (labelled SMS data)         |
| Deployment        | Streamlit app (`app.py`)                    |

---

## 📁 Folder Structure

sms-spam-classifier/ <br>
│
├── app.py # Streamlit app script <br>
├── model.pkl # Trained ML model<br>
├── vectorizer.pkl # TF-IDF vectorizer<br>
├── sms-spam-detection.ipynb# Jupyter notebook (model training)<br>
├── sms_spam.csv # Dataset<br>
├── requirements.txt # Project dependencies<br>
└── README.md # Project documentation<br>


---

## 📊 Dataset

The dataset `sms_spam.csv` contains SMS messages labeled as `spam` or `ham` (not spam).

| Column     | Description                  |
|------------|------------------------------|
| label      | Spam or Ham                  |
| message    | Text of the SMS message      |

---

## 🔎 Preprocessing Steps

- Lowercasing text
- Removing punctuation & special characters
- Tokenization
- Stopword removal
- Vectorization using TF-IDF

---

## 📈 Model Training

- Model: Naive Bayes
- Evaluation Metric: Accuracy, Precision, Recall
- Training done in `sms-spam-detection.ipynb`

🙋‍♂️ Author
Jay Devmurari
📧 devmurarijay66@gmail.com

