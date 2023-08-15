import streamlit as st
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import pickle

ps = PorterStemmer()




# Transforming Text function
def transform_text(text):   
    # lowering the characters
    text = text.lower()
    
    # sparating words
    text = nltk.word_tokenize(text)
    
    # removing special charcters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
     
    # removing stopwords and punctuations in text
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]  # cloning as list is mutuable data type
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
     
    # returning as a string rather words
    return " ".join(y)




tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('Email / SMS Spam Classifier')
input_sms  = st.text_area("Enter the Email/Message")

if st.button('Predict'):


     # pipeline for oue model

     # 1) preprocess
     transformed_sms = transform_text(input_sms)

     # 2) vectorize
     vector_input = tfidf.transform([transformed_sms])

     # 3) predict
     result = model.predict(vector_input)

     # 4) display
     if result == 0:
          st.header('Not Spam')
     else:
          st.header('Spam')