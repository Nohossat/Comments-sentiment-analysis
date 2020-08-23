# Flask and utils
from flask import Flask, url_for, request, render_template, redirect
from flask_caching import Cache
import numpy as np 
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import FrenchStemmer
from stop_words import get_stop_words
import string
import re
import pickle
import pandas as pd

# Scikit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import make_pipeline


nltk.download('punkt')

app = Flask(__name__) 

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 400
}

# tell Flask to use the above defined config
app.config.from_mapping(config)

def remove_stopwords(commentaire, stop_words):
    # remove stop words, punctuation and words which length is below 2, numbers and none values
    ponctuations = string.punctuation
    p = re.compile(r'\.+')

    commentaire = [p.sub(r'', word).lower() for word in commentaire if word.lower() not in stop_words and word not in ponctuations and not word.isnumeric() and (len(word) > 2 or word == 'ne')]
    return ' '.join(commentaire)

def stem_review(review):
    stem = FrenchStemmer()
    review = review.split(' ')
    return [stem.stem(word) for word in review]

def prepare_test_data(test_review):
    # on doit prendre en compte la négation
    stop_words = get_stop_words('french').copy()
    stop_words.remove('ne')
    stop_words.remove('pas')

    # tokenize data
    review_tokens = word_tokenize(test_review)
    
    # remove stop words
    review_tokens = remove_stopwords(review_tokens, stop_words)
    
    #stem tokens
    review_tokens = stem_review(review_tokens)
    
    #return string
    return ' '.join(review_tokens)

def get_model():
    clf = pickle.load("models/model.pkl")
    # get points
    pipe = make_pipeline(CountVectorizer(),
                        TfidfTransformer())

    return clf, pipe

def fit_model():
    df = pd.read_csv('static/data/dataset_note_booking.csv')
    
    df = df.dropna()

    # split data
    X_train, X_test, y_train, y_test = train_test_split(df["review"], df['polarite'], test_size=0.2, random_state=0)
    
    # get points
    pipe = make_pipeline(CountVectorizer(),
                        TfidfTransformer())

    feat_train = pipe.fit_transform(X_train)
    feat_test = pipe.transform(X_test)

    # train model
    clf = LogisticRegression(random_state=0, solver='newton-cg')
    clf.fit(feat_train, y_train)
    score_train = np.mean(cross_val_score(clf, feat_train, y_train, cv=5))
    score_test = np.mean(cross_val_score(clf, feat_test, y_test, cv=5)) 
    y_pred = clf.predict(feat_train)
    f1score = f1_score(y_train, y_pred)

    print(score_train, score_test, f1score) 

    return clf, pipe

def predict_comment(test_review, clf, pipe):  
    # prepare data
    review = prepare_test_data(test_review)
    print(review)

    # predict 
    transformed_review = pipe.transform([review])
    
    predict = "Négatif" 

    print(clf.predict(transformed_review))

    if clf.predict(transformed_review):
        predict = "Positif"

    return predict

def save_comment(new_comment, df):
    df.loc[len(df.index)] = new_comment
    df.to_csv('static/data/comments.csv') 
 
clf, pipe = get_model() 

@app.route('/')
def index(prestations=None, titre=None, desc=None, comments=None):
    # get prestations of simplon hotel
    prestations = [    
        {
            'icon': 'fa fa-home', 
            'desc': '20 villas',
        },
        {
            'icon': 'fa fa-cutlery',
            'desc': 'Restaurant gastronomique',
        },
        {
            'icon': 'fas fa-umbrella-beach',
            'desc': 'Plage privée',
        },
        {
            'icon': 'fas fa-spa',
            'desc': 'Spa services'
        }
    ]

    # read the database to fetch comments
    df_comments = pd.read_csv('static/data/comments.csv', index_col='Unnamed: 0')

    # we want to fead the latest comments first
    comments = df_comments.values[::-1]  

    #nb_comments in database
    nb_comments = len(df_comments.index)

    description_hotel = "Vous venez d'ouvrir un hôtel. \
    Comme vous n'êtes pas sûr de la qualité de votre établissement, \
    vous permettez aux personnes de poster des commentaires mais pas de mettre de note. \
    Cependant, vous voulez quand même déterminer si le commentaire est positif ou négatif. \
    Pour cela, vous allez scrapper des commentaires sur booking et leur note associée afin de \
    faire tourner un algorithme de classification pour faire des prédictions sur vos propres commentaires."
    titre = 'Booking Sentiment Analysis'

    return render_template('home.html', 
                            prestations=prestations, 
                            titre= titre, 
                            desc=description_hotel, 
                            comments=comments,
                            nb_comments = nb_comments) 

@app.route('/create_comment', methods=['POST'])
def create_comment(titre=None, parts=None, objectif=None, variables=None, clf=clf, pipe=pipe):
    
    nom = "Anonyme"
    if request.form['nom_user']:
        nom = request.form['nom_user']

    comment = request.form['comment']

    # predict if comment is positive or negative
    result = predict_comment(comment, clf, pipe)

    # save to database
    df_comments = pd.read_csv('static/data/comments.csv')
    save_comment([nom, comment, result], df_comments)

    # refresh homepage and go straight to the comments section
    return redirect(url_for('index') + '#comments') 

if __name__ == '__main__':
    app.run(debug=True)
