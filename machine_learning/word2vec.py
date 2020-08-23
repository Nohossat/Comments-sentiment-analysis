from gensim.models import Word2Vec
import numpy as np


# load normalized data
normalized_tokens = np.load('../datasets/normalized_tokens.npy', allow_pickle=True)


def get_word_embedding(corpus):
    model = Word2Vec(corpus, size=5, workers=6, iter=10, min_count=1)
    model.save("../models/word2vec.bin")
    return model

def embed_corpus(corpus, max_length_doc=None):
    model = Word2Vec.load("models/word2vec.bin")

    word_embedding = []

    if max_length_doc is None:
        max_length_doc = max([len(sentence.split(' ')) for sentence in corpus])

    for sentence in corpus:
        tokens = sentence.split(' ')
        if len(tokens) < max_length_doc:
            padding_nb = max_length_doc - len(tokens)
            tokens.extend(["hotel"]* padding_nb)
        word_embedding.append(model.wv[tokens].flatten())

    return word_embedding


if __name__ =='__main__':
    model = get_word_embedding(normalized_tokens)
    model1 = Word2Vec.load("../models/word2vec.bin")
    print(model1.wv[["qualite/prix", "superb"]])


