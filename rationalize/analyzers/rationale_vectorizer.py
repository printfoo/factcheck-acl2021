# coding: utf-8


import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


class Tagger(object):
    """
    Dataset Tagger for movie reviews.
    Add rationales to train.tsv, dev.tsv and test.tsv.
    """


    def __init__(self, damp):
        self.data_dirs = ["train.tsv", "dev.tsv", "test.tsv"]
        self.rationale_dir = "soft_rationalizer.analyze"
        self.stopwords = set(stopwords.words("english")).union({"movie", "film"})
        self.word2vec = self._get_word2vec()
        self.damp = damp

    
    def _get_word2vec(self):
        word2vec = {}
        with open(os.path.join("..", "glove", "glove.6B.200d.txt")) as f:
            word2vec_lines = f.read().split("\n")
        for word2vec_line in word2vec_lines:
            word2vec_line = word2vec_line.split(" ")
            word = word2vec_line[0]
            vec = [float(_) for _ in word2vec_line[1:]]
            word2vec[word] = np.array(vec)
        for word in self.stopwords:
            word2vec[word] = np.zeros(200)
        return word2vec
        
        
    def _get_hard_from_soft(self, soft):
        hard = [0] * len(soft)
        max_val = max(soft)
        
        while True:
            this_max_id = np.argmax(soft)
            this_max_val = soft[this_max_id]
        
            if this_max_val < max_val * self.damp:
                break
            
            hard[this_max_id] = 1  # Update hard.
            soft[this_max_id] = 0  # Clear soft.
            
            # Adding left neighbors.
            this_id = this_max_id - 1
            while this_id > -1 and soft[this_id] > this_max_val * self.damp:
                hard[this_id] = 1  # Update hard.
                soft[this_id] = 0  # Clear soft.
                this_id -= 1
            
            # Adding right neighbors.
            this_id = this_max_id + 1
            while this_id < len(soft) and soft[this_id] > this_max_val * self.damp:
                hard[this_id] = 1  # Update hard.
                soft[this_id] = 0  # Clear soft.
                this_id += 1

        return hard


    def _get_rationale(self, row):
        tokens = row["tokens"].split(" ")
        soft_rationale = [float(_) for _ in row["soft_rationale"].split(" ")]
        while soft_rationale[-1] == 0:
            soft_rationale.pop()
        assert len(soft_rationale) == len(tokens), "Error in length!"
        hard_rationale = self._get_hard_from_soft(soft_rationale)
        row["rationale_len"] = sum(hard_rationale)
        rationale_phrases = [""]
        for i, r in enumerate(hard_rationale):
            if (not r and tokens[i] == ".") and rationale_phrases[-1] != "":
                rationale_phrases[-1] = rationale_phrases[-1].strip(" ")
                rationale_phrases.append("")
            elif r:
                rationale_phrases[-1] += tokens[i] + " "
        row["rationale_phrases"] = rationale_phrases[:-1]
        return row
    
    
    def _get_embedding(self, row):
        embedding = np.zeros(200)
        for word in row["rationale"].split(" "):
            if word not in self.word2vec:
                return np.nan
            embedding += self.word2vec[word]
        embedding /= len(row["rationale"].split(" "))
        if sum(embedding) == 0:
            return np.nan
        return " ".join([str(e) for e in embedding])


    def _count_rationale(self, rationale_phrases_all):
        count = {}
        for rationale_phrases in rationale_phrases_all:
            for rationale_phrase in rationale_phrases:
                if rationale_phrase not in count:
                    count[rationale_phrase] = 0
                count[rationale_phrase] += 1
        return count

    
    def tag(self, df):
        df = df.apply(self._get_rationale, axis=1)
        pos = self._count_rationale(df[df["label"] == "positive"]["rationale_phrases"].tolist())
        pos = pd.DataFrame.from_dict(pos, orient="index", columns=["count"])
        pos["label"] = "positive"
        neg = self._count_rationale(df[df["label"] == "negative"]["rationale_phrases"].tolist())
        neg = pd.DataFrame.from_dict(neg, orient="index", columns=["count"])
        neg["label"] = "negative"
        df = pd.concat([pos, neg])
        df = df.sort_values("count", ascending=False)
        df["rationale"] = df.index
        df["embeddings"] = df.apply(self._get_embedding, axis=1)
        return df


def vectorize(analyze_out, train_args):
    print("Hi")
    exit()
    tagger = Tagger(0.7)
    dfs = []
    for i in range(3):
        df = pd.read_csv(tagger.data_dirs[i], sep="\t")
        df["index"] = df.index
        rationale = pd.read_csv(os.path.join(tagger.rationale_dir, tagger.data_dirs[i]), sep="\t")
        df = df.merge(rationale, on="index")
        dfs.append(df)
    df = pd.concat(dfs)
    df = tagger.tag(df)
    df.to_csv("rationale_embeddings.csv", index=False)
