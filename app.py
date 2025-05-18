# import streamlit as st
# import pickle
# import string
#
# from nltk.stem.porter import PorterStemmer
# import nltk
#
# # Ensure punkt is available when the app starts
# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')
#
# nltk.download('stopwords')
#
#
# from nltk.corpus import stopwords
#   # Correct resource
# ps = PorterStemmer()
# def transform_text(text):
#     text = text.lower()
#
#     text = nltk.word_tokenize(text)  # tokenizer will separate each words
#
#     y = []
#     for i in text:  # removing special chars
#         if i.isalnum():
#             y.append(i)
#
#     text = y[:]
#     y.clear()
#
#     for i in text:  # removing stop words and punctuation marks
#         if i not in stopwords.words('english') and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()
#
#     for i in text:
#         y.append(ps.stem(i))
#
#     return " ".join(y)
#
# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))
#
# st.title("Email/SMS Spam Classifier")
#
# input_sms = st.text_area("Enter The Message")
#
# if st.button('Predict'):
#     # preprocess
#     transform_sms = transform_text(input_sms)
#
#     # vectorize
#     vector_input = tfidf.transform([transform_sms])
#
#     # predict
#     result = model.predict(vector_input)[0]
#
#     # display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")


import streamlit as st
import pickle
import string
import nltk

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download NLTK resources only if not already present
def download_nltk_resources():
    for resource in ['punkt', 'stopwords']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
        except LookupError:
            nltk.download(resource)

download_nltk_resources()

ps = PorterStemmer()

def transform_text(text):
    # Lowercase
    text = text.lower()

    # Tokenize
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# Load vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Model or vectorizer file not found: {e}")
    st.stop()

# Streamlit UI
st.title("📩 Email/SMS Spam Classifier")

input_sms = st.text_area("✉️ Enter the message here:")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message before predicting.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display
        st.success("✅ Not Spam" if result == 0 else "🚫 Spam")
