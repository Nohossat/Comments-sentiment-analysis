from gensim.models import Word2Vec
import numpy as np


# load normalized data
normalized_tokens = np.load('datasets/normalized_tokens.npy', allow_pickle=True)


def get_word_embedding(corpus):
    model = Word2Vec(corpus, size=10, workers=6, iter=10, min_count=1)
    model.save("models/word2vec.bin")
    return model


if __name__ =='__main__':
    model = get_word_embedding(normalized_tokens)
    model = Word2Vec.load("models/word2vec.bin")
    print(model.wv[["qualite/prix", "superb"]])


