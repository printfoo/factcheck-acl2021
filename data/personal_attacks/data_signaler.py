# coding: utf-8


import os, json
import pandas as pd
from scipy.special import softmax


class DataSignaler(object):
    """
    Dataset signaler for personal attacks.
    Add signals to train.tsv.
    """

    def __init__(self):
        self.train_dir = "train.tsv"
        self.vocab_dir = os.path.join("linear_bow.analyze", "word_weight.json")
        self.NEG_INF = -1.0e6
        
    
    def _get_signal(self, row):
        signal_dict = self.signal_dicts[row["label"]]
        comment = row["comment"].split(" ")
        signal = [signal_dict[c] if c in signal_dict else self.NEG_INF for c in comment]
        signal = ["{:.5f}".format(s) for s in softmax(signal)]
        return " ".join(signal)
            
    
    def signal(self):
        df = pd.read_csv(self.train_dir, sep="\t")
        vocab = pd.read_json(self.vocab_dir, lines=True)
        labels = set(df["label"].tolist())
        self.signal_dicts = {}
        for l in labels:
            self.signal_dicts[l] = {w: s for w, s in vocab[vocab[l] > 0][["word", l]].values}

        df["signal"] = df.apply(self._get_signal, axis=1)
        df.to_csv(self.train_dir, index=False, sep="\t")


if __name__ == "__main__":
    DataSignaler().signal()
    
