import pickle
import tensorflow as tf 

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
#text preprocessing
ps = PorterStemmer()

def preprocess(line):
    review = re.sub('[^a-zA-Z]', ' ', line) #leave only characters from a to z
    review = review.lower() #lower the text
    review = review.split() #turn string into list of words
    #apply Stemming 
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] #delete stop words like I, and ,OR   review = ' '.join(review)
    #trun list into sentences
    return " ".join(review)

print("""
# Emotion detection app

This app detects the emotion of a given text

Data obtained from Kaggle: https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp.
""")

print('Enter your sentence below: ')

input_text = input()

# Load the encoder and model
encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))
model = tf.keras.models.load_model('my_model.h5')

if input_text:
    # Preprocess the input text (if necessary)
    # For this example, we'll skip preprocessing
    input_text = preprocess(input_text)
    # Transform input text into a vector using CountVectorizer
    array = cv.transform([input_text]).toarray()

    # Predict the emotion
    pred = model.predict(array)
    a = pred.argmax(axis=1)
    prediction = encoder.inverse_transform(a)[0]

    print('Prediction:', prediction)
