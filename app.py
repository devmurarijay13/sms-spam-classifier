import streamlit as st
import pickle
import string

from nltk.stem.porter import PorterStemmer
import nltk

# Ensure punkt is available when the app starts
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

nltk.download('stopwords')


from nltk.corpus import stopwords
  # Correct resource
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()

    text = nltk.word_tokenize(text)  # tokenizer will separate each words

    y = []
    for i in text:  # removing special chars
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:  # removing stop words and punctuation marks
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter The Message")

if st.button('Predict'):
    # preprocess
    transform_sms = transform_text(input_sms)

    # vectorize
    vector_input = tfidf.transform([transform_sms])

    # predict
    result = model.predict(vector_input)[0]

    # display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")