# coding: utf-8


from gensim.models import Word2Vec
import os, json, random
import pandas as pd
import numpy as np
np.random.seed(0)
random.seed(0)


# Read data.
corpus = []
for set_name in ["train", "dev", "test"]:
    df = pd.read_csv(set_name + ".tsv", sep="\t")
    df["tokens"] = df["tokens"].apply(lambda t: t.split(" "))
    corpus.extend(df["tokens"].tolist())


# Train w2v.
model = Word2Vec(corpus, vector_size=200, window=5, min_count=1, workers=5)


# Save w2v.
model.wv.save_word2vec_format("w2v.txt", write_header=False)
