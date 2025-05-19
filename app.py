import streamlit as st
import pickle
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Setup NLTK data directory
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path.append(nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, "tokenizers", "punkt")):
    nltk.download("punkt", download_dir=nltk_data_path)
if not os.path.exists(os.path.join(nltk_data_path, "corpora", "stopwords")):
    nltk.download("stopwords", download_dir=nltk_data_path)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stopwords.words('english') and w not in string.punctuation]
    words = [ps.stem(w) for w in words]
    return " ".join(words)

# Load model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message:")

if st.button("Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
