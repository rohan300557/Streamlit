import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from pickle import dump, load

classifier_loc = "pickle/logit_model.pkl"
encoder_loc = "pickle/countvectorizer.pkl"
image_loc = "data/spam-filter.png"


def preprocess(message):
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    clean_msg_lst = []
    nonpunc = [c for c in message if c not in string.punctuation]
    nonpunc = "".join(nonpunc)
    clean_mess = [word.lower() for word in nonpunc.split() if word.lower() not in stopwords.words("english")]
    token = [word_tokenize(word) for word in nonpunc.split()]
    token = [''.join(ele) for ele in token]
    clean_mess = [stemmer.stem(word) for word in token]
    clean_msg_lst.append(" ".join(clean_mess))
    return (clean_msg_lst[0])



def predict(message):
    
    vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))

    classifier = load(open('pickle/logit_model.pkl', 'rb'))

    clean_message = preprocess(message)

    message_encoded = vectorizer.transform([clean_message])

    message_input = message_encoded.toarray()

    prediction = classifier.predict(message_input)

    return prediction



def main():
    st.title('Spam-Ham Predictor')

    st.sidebar.subheader("Spam ham Predictor")

    message = st.text_input('Enter your Message')

    prediction = predict(message)

    if(message):
        st.subheader("Prediction:")
        if(prediction == 0):
            st.image("data/ham.jpg", use_column_width = True)
            # st.write("Ham :smile:")
        else:
            st.image("data/spam.jpg", use_column_width = True)
            # st.write("Spam :cry:")



if(__name__ == '__main__'):
    main()