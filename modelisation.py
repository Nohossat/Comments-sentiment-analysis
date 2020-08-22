# Utils
import pandas as pd
import numpy as np
import time
import pickle

# Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score, auc

# NLP
from gensim.models import Word2Vec

# XGBoost
from xgboost.sklearn import XGBClassifier

# Analyse comparative des modèles de classification
    

def run_model(model_name, train_name, model, X_train, y_train, default_params = None, grid_search_params = None):
    '''
    run model with different feature transformations + with or without GridSearch

    Parameters : 
    model_name : the name of the model saved in the CSV file
    train_name : name of the x_train set
    model : model used for training
    X_train : training X data
    y_train : training label
    default_params : 
    grid_search_params : 

    Output :
    None - the results are saved into a CSV file
    '''

    # measures the time taken by the model to run
    start_time = time.time()

    # get model
    if default_params:
        clf = model(**default_params)
    else : 
        clf = model()
    
    # fit model with or without GridSearchCV
    if grid_search_params :
        grid_clf = GridSearchCV(clf, grid_search_params, n_jobs=-1)
        clf = grid_clf.fit(X_train, y_train)
        best_params = clf.best_params_
        print(best_params)
    else:
        clf.fit(X_train, y_train)
        best_params = None
            
    # save model
    pickle.dump(clf, f'models/{model_name}.pkl')

    # get predictions
    y_pred = clf.predict(X_train)
    
    # we will record some metrics in a CSV file for presentation
    duration = time.time() - start_time
    accuracy = np.mean(cross_val_score(clf, X_train, y_train, cv=5)) # avg score on the CV
    precision = precision_score(y_train, y_pred, average=None)
    recall = recall_score(y_train, y_pred, average=None)
    f1score = f1_score(y_train, y_pred, average=None)
    results = [[ model_name, train_name, accuracy, precision, recall, f1score, best_params, duration ]]
    
    # save to csv
    cols = ['model_name', 'feature transformation', 'accuracy', 'precision', 'recall', 'f1_score', 'best paramaters', 'duration']
    
    try:
        backup = pd.read_csv('metrics/results.csv')
    except Exception as e:
        print(e)
        backup = pd.DataFrame([], columns=cols)
    
    model_metrics = pd.DataFrame(results, columns=cols)
    
    backup = pd.concat([backup, model_metrics])
    backup.to_csv('metrics/results.csv', index=False)
    
    return backup

def get_models_results(dataset):
    '''
    run all predefined models and paramaters and save the results into a CSV file

    Parameters : 
    dataset : the dataset used for the modeling

    Output :
    None -the results are saved into a CSV file
    '''
    
    # get split data
    X = dataset['review'].values
    y = dataset['note'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle = True, random_state=0)
    
    # get different types of X_train according to feature transformation
    train_data = {}
    
    # Tf-idf Vectorizer
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    train_data['TF-IDF'] = X_train_tfidf
    
    
    # CountVectorizer + N-gram + TF-IDF
    pipe_ngram = make_pipeline(CountVectorizer(min_df=0.0005, ngram_range=(1, 2)), TfidfTransformer())
    X_train_ngram = pipe_ngram.fit_transform(X_train).toarray()
    train_data['CV(n-gram) + TF-IDF'] = X_train_ngram
    
    
    # TF-IDF Truncated SVD
    pipe_svd_tfidf = make_pipeline(TfidfVectorizer(), TruncatedSVD(n_components=300))
    X_train_svd = pipe_svd_tfidf.fit_transform(X_train)
    train_data['CV + TF-IDF + SVD'] = X_train_svd

    # Word2Vec
    model = Word2Vec.load("models/word2vec.bin")
    X_train_wv = [ model.wv[sentence.split(' ')] for sentence in X_train ] 
    train_data['Word2Vec'] = X_train_wv
    
    # list models
    models = {
        'Regression logistique l1' : LogisticRegression,
        'Regression logistique l2' : LogisticRegression,
        'Regression logistique Elastic Net' : LogisticRegression,
        'NB : Naive Bayes' : MultinomialNB, 
        #'Random Forest' : RandomForestClassifier,
        #'XGB': XGBClassifier,
        #'SVC' : SVC,
        #'AdaBoost': AdaBoostClassifier
    }
    
    # related params for GridSearch and random_state
    params = {
        'Regression logistique l1' : [{'random_state' : 0}, {'penalty' : ['l1'], 'solver': ['saga', 'liblinear'], 'C': [1.0, 10.0, 50.0], 'n_jobs' : [-1]}],
        'Regression logistique l2' : [{'random_state' : 0}, {'penalty' : ['l2'], 'solver': ['saga', 'sag', 'newton-cg', 'lbfgs'], 'C': [1.0, 10.0, 50.0], 'n_jobs' : [-1]}],
        'Regression logistique Elastic Net' : [{'random_state' : 0}, {'penalty' : ['elasticnet'], 'solver': ['saga'], 'l1_ratio' : [0.2, 0.5, 0.8], 'n_jobs' : [-1]}],
        'NB : Naive Bayes' : [ None, None ],
        'Random Forest' : [{'random_state' : 0}, 
                            {'boostrap' : [True], 
                            'criterion' : ['gini'], 
                            'n_estimators' : [50, 100, 200],
                            'max_depth': [6, 30, 50],
                            'n_jobs' : [-1]}], 
        'XGB': [{'random_state' : 0}, 
                {'learning_rate' : [0.05, 0.01, 0.2],
                 'max_depth' : [6, 30, 50],
                 'n_estimators' : [50, 200], 
                 'n_jobs':[-1]}],
        'SVC' : [{'random_state' : 0}, 
                { 'C': [1, 5, 10, 50],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005],
                 'kernel': ['rbf','sigmoid']}, 
                 ],
        'AdaBoostClassifier' : [ {'random_state' : 0}, 
                                {'base_estimator' : [MultinomialNB, SVC], 'n_estimators' : [50, 200]}
                                ]
    }

    params1 = {
        'Regression logistique l1' : [{'random_state' : 0}, {'penalty' : ['l1'], 'solver': ['saga', 'liblinear'], 'C': [1.0, 10.0, 50.0], 'n_jobs' : [-1]}],
        'Regression logistique l2' : [{'random_state' : 0}, {'penalty' : ['l2'], 'solver': ['saga', 'sag', 'newton-cg', 'lbfgs'], 'C': [1.0, 10.0, 50.0], 'n_jobs' : [-1]}],
        'Regression logistique Elastic Net' : [{'random_state' : 0}, {'penalty' : ['elasticnet'], 'solver': ['saga'], 'l1_ratio' : [0.2, 0.5, 0.8], 'n_jobs' : [-1]}],
        'NB : Naive Bayes' : [ None, None ]
    }
    
    # run models with different parameters and different feature extraction methods
    for name_model, model in models.items():
        for name_X_train, X_train in train_data.items():
            #get metrics for train data
            try : 
                run_model(name_model, 
                          f'{name_X_train}_train', 
                          model, 
                          X_train, 
                          y_train, 
                          default_params = params1[name_model][0], 
                          grid_search_params = params1[name_model][1])
            except Exception as e:
                print(e)
            print('training: done')
        
            # get metrics for test data
            try : 
                run_model(name_model, 
                        f'{name_X_train}_test', 
                        model, 
                        X_test, 
                        y_test, 
                        default_params = params1[name_model][0], 
                        grid_search_params = params1[name_model][1])
            except Exception as e:
                print(e)
            print('testing: done')

if __name__ == '__main__':
    dataset_note_booking = pd.read_csv("datasets/processed_data.csv")

    # on récupère tous les modèles et leurs métriques dans le fichier CSV booking_models_metrics.csv
    get_models_results(dataset_note_booking)
