from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import streamlit as st

#step 1
word_index=imdb.get_word_index()
reverse_word_index=dict([(value,key) for (key,value) in word_index.items()])

model=load_model('simple_rnn_imdb.h5')

##step 2

def decoded_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in sample_review])

def preprocess_text(text):
  words=text.lower().split()
  encoded_review=[word_index.get(word,0) for word in words]
  padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
  return padded_review


#step 3
## streamlit app

st.title('IMDB movie review sentiment analysis')
st.write('Enetr a movie review to classify it as positive or negative.')

#user input

user_input=st.text_area('movie review')

if st.button('Classify'):
  preproccess_input=preprocess_text(user_input)
  prediction=model.predict(preproccess_input)
  sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]}')
else:
  st.write('Please enter a movie review.')  

