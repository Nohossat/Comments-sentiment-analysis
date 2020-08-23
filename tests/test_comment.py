from hotel_app.comment import Comment
import os
import pickle
import pytest
import pandas as pd

FILE_REL_PATH = os.path.relpath(__file__)
CURRENT_DIR = os.path.dirname(FILE_REL_PATH)

@pytest.fixture
def comment():
    '''init a comment with a negative comment'''
    return Comment("C'est la pire expérience que j'ai eu dans un hotel, c'était médiocre", "Antoine")

def test_prepare_test_data(comment):
    comment.prepare_test_data()
    assert comment.transformed_comment == "c'est pir expérient j'ai hotel c'et médiocr"

def test_save_comment(comment):
    df_path = "data/comments.csv"
    comment.result = "Négatif"
    comment.save_comment(df_path)
    df = pd.read_csv(df_path)
    assert (df.iloc[len(df.index) - 1].values == [ "Antoine", "C'est la pire expérience que j'ai eu dans un hotel, c'était médiocre", "Négatif"]).all(), "The values are incorrect"

@pytest.mark.parametrize("author, document, result", [
    ("Marie", "J'ai adoré mon séjour", "Positif"),
    ("Anonyme", "Hôtel sale, petit déjeuner inexistant", "Négatif"),
    ("Julie", "C'est la pire expérience que j'ai eu dans un hotel", "Négatif")
])
def test_comments(comment, author, document, result):
    # change author and document values
    comment.author = author
    comment.document = document

    # get model and transformer pipe
    clf_path = '../models/Random Forest_TF-IDF_train.pkl'
    pipe_path = '../models/tf_idf_pipeline.pkl'

    clf = pickle.load(open(clf_path, "rb"))
    pipe = pickle.load(open(pipe_path, "rb"))

    comment.predict_comment(clf, pipe)
    assert comment.result == result