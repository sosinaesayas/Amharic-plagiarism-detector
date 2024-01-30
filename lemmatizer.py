import os
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('wordnet')

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])