# coding: utf-8


import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


class Vectorizer(object):
    """
    Get vectors for rationales.
    """


    def __init__(self, data_path, rationale_path, vector_path, train_args):
        dfs = []
        for set_name in ["train", "dev", "test"]:
            data = pd.read_csv(os.path.join(data_path, set_name + ".tsv"), sep="\t")
            rationale = pd.read_csv(os.path.join(rationale_path, set_name + ".tsv"), sep="\t")
            data["index"] = data.index
            data = data.merge(rationale, on="index")
            dfs.append(data)
        self.df = pd.concat(dfs)
        self.vector_path = vector_path
        self.stopwords = set(stopwords.words("english")).union({"movie", "film"})
        self.threshold = train_args.binarize_threshold

    
    def get_word2vec(self, embedding_dim, embedding_path):
        word2vec = {}
        with open(embedding_path) as f:
            word2vec_lines = f.read().split("\n")
        for word2vec_line in word2vec_lines:
            word2vec_line = word2vec_line.split(" ")
            word = word2vec_line[0]
            vec = [float(_) for _ in word2vec_line[1:]]
            word2vec[word] = np.array(vec)
        for word in self.stopwords:
            word2vec[word] = np.zeros(embedding_dim)
        self.word2vec = word2vec
        

    def _get_hard_from_soft(self, soft, damp=0.7):  # Old rule.
        hard = [0] * len(soft)
        max_val = max(soft)
        
        while True:
            this_max_id = np.argmax(soft)
            this_max_val = soft[this_max_id]
        
            if this_max_val < max_val * damp:
                break
            
            hard[this_max_id] = 1  # Update hard.
            soft[this_max_id] = 0  # Clear soft.
            
            # Adding left neighbors.
            this_id = this_max_id - 1
            while this_id > -1 and soft[this_id] > this_max_val * damp:
                hard[this_id] = 1  # Update hard.
                soft[this_id] = 0  # Clear soft.
                this_id -= 1
            
            # Adding right neighbors.
            this_id = this_max_id + 1
            while this_id < len(soft) and soft[this_id] > this_max_val * damp:
                hard[this_id] = 1  # Update hard.
                soft[this_id] = 0  # Clear soft.
                this_id += 1

        return hard
    
    
    def _binarize_rationale(self, rationale_pred):
        return [float(_ > self.threshold) for _ in rationale_pred]


    def _get_rationale(self, row):
        tokens = row["tokens"].split(" ")
        rationale_pred = [float(r) for r, m in zip(row["rationale_pred"].split(" "), row["mask"].split(" ")) if float(m) > 0]
        assert len(rationale_pred) == len(tokens), "Error in length!"
        rationale_binary = self._binarize_rationale(rationale_pred)
        row["rationale_len"] = sum(rationale_binary)
        rationale_phrases = [""]
        for i, r in enumerate(rationale_binary):
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

    
    def vectorize(self):
        df = self.df.apply(self._get_rationale, axis=1)
        labels = set(df["label"])
        dfs = []
        for label in labels:
            label_df = self._count_rationale(df[df["label"] == label]["rationale_phrases"].tolist())
            label_df = pd.DataFrame.from_dict(label_df, orient="index", columns=["count"])
            label_df["label"] = label
            dfs.append(label_df)
        df = pd.concat(dfs)
        df = df.sort_values("count", ascending=False)
        df["rationale"] = df.index
        df["embeddings"] = df.apply(self._get_embedding, axis=1)
        df.to_csv(os.path.join(self.vector_path, "rationale_embeddings.csv"), index=False)


def vectorize(data_path, rationale_path, vector_path, train_args):
    
    if not os.path.exists(vector_path):
        os.mkdir(vector_path)

    vectorizer = Vectorizer(data_path, rationale_path, vector_path, train_args)
    vectorizer.get_word2vec(train_args.embedding_dim, train_args.embedding_dir)
    vectorizer.vectorize()
