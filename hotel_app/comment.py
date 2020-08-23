# Utils
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from stop_words import get_stop_words
import string
import re
import pandas as pd
import os

FILE_REL_PATH = os.path.relpath(__file__)
CURRENT_DIR = os.path.dirname(FILE_REL_PATH)  # get directory path of file


class Comment():
    def __init__(self, document, author):
        self.document = document
        self.author = author
        self.transformed_comment = ''
        self.result = None

    def remove_stopwords(self, stop_words):
        # remove stop words, punctuation and words which length is below 2, numbers and none values
        ponctuations = string.punctuation
        p = re.compile(r'\.+')

        self.transformed_comment = [p.sub(r'', word).lower() for word in self.transformed_comment if word.lower() not in stop_words and word not in ponctuations and not word.isnumeric() and len(word) > 2]
        self.transformed_comment = ' '.join(self.transformed_comment)

    def stem_review(self):
        stem = FrenchStemmer()
        self.transformed_comment = self.transformed_comment.split(' ')
        self.transformed_comment = [stem.stem(word) for word in self.transformed_comment]
        self.transformed_comment = ' '.join(self.transformed_comment)

    def prepare_test_data(self, lang="french"):
        if lang == "french":
            stop_words = get_stop_words('french').copy()
            stop_words.remove('ne')
            stop_words.remove('pas')
        else : # english
            stop_words = get_stop_words('english').copy()

        # tokenize data
        self.transformed_comment = word_tokenize(self.document)
        
        # remove stop words
        self.remove_stopwords(stop_words)
        
        # stem tokens
        self.stem_review()
        
    def predict_comment(self, clf, pipe):  
        # prepare data
        self.prepare_test_data()

        # transform data before predictions
        transformed_review = pipe.transform([self.transformed_comment])
        
        # predict polarity
        predict = "Négatif" 
        if clf.predict(transformed_review):
            predict = "Positif"

        self.result = predict

    def save_comment(self, df):
        df.loc[len(df.index)] = [self.author, self.document, self.result]
        comments_csv = os.path.join(CURRENT_DIR, 'static/data/comments.csv')
        df.to_csv(comments_csv, index=False) 
    