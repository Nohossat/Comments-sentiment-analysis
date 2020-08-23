# Flask
from flask import Flask, url_for, request, render_template, redirect
from hotel_app import app

# Utils
import numpy as np 
import os
import pickle
import pandas as pd
from hotel_app.comment import Comment

# Scikit Learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import make_pipeline


config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "simple", # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 400
}

FILE_REL_PATH = os.path.relpath(__file__)
CURRENT_DIR = os.path.dirname(FILE_REL_PATH)  # get directory path of file
PROJECT_DIR = os.path.dirname(CURRENT_DIR)  # get directory path of file

# tell Flask to use the above defined config
app.config.from_mapping(config)

# get model and transformer pipe
clf_path = os.path.join(PROJECT_DIR, 'models/Random Forest_TF-IDF_train.pkl')
pipe_path = os.path.join(PROJECT_DIR, 'models/tf_idf_pipeline.pkl')

clf = pickle.load(open(clf_path, "rb"))
pipe = pickle.load(open(pipe_path, "rb"))

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
    comments_csv = os.path.join(CURRENT_DIR, 'static/data/comments.csv')
    df_comments = pd.read_csv(comments_csv)

    # we want to fetch the latest comments first
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
    # get author name
    nom = "Anonyme"
    if request.form['nom_user']:
        nom = request.form['nom_user']

    # get comment
    comment = Comment(request.form['comment'], nom)

    # predict if comment is positive or negative
    comment.predict_comment(clf, pipe)

    # save to database
    comments_csv_path = os.path.join(CURRENT_DIR, 'static/data/comments.csv')
    comment.save_comment(comments_csv_path)

    # refresh homepage and go straight to the comments section
    return redirect(url_for('index') + '#comments') 
