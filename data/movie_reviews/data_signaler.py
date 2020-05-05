# coding: utf-8


import os, json
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer


class DataSignaler(object):
    """
    Dataset signaler for movie reviewss.
    Add linear signal and domain knowledge to train.tsv, dev.tsv and test.tsv.
    """

    def __init__(self):
        self.data_dirs = ["train.tsv", "dev.tsv", "test.tsv"]
        
        # Linear signal
        self.vocab_dir = os.path.join("linear_bow.analyze", "word_weight.json")
        vocab = pd.read_json(self.vocab_dir, lines=True)
        self.signal_dicts = {}
        for l in set(vocab.columns) - {"word"}:
            self.signal_dicts[l] = {w: s for w, s in vocab[["word", l]].values}
            print(l, "fuck", self.signal_dicts[l]["fuck"])
        signal_sorted = sorted(self.signal_dicts[l].items(), key=lambda _: abs(_[1]))
        threshold = signal_sorted[int(len(signal_sorted) * -0.05):]
        print("Threshold:", threshold[0][1])
        
        # Domain knowledge.
        self.vocab_dir = os.path.join("raw", "NRC-Emotion-Lexicon-v0.92",
                                      "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
        with open(self.vocab_dir, "r") as f:
            vocab = f.read().strip().lower().split("\n")
        vocab = [_.split("\t") for _ in vocab]
        self.domain_dicts = {}
        for l in self.signal_dicts:
            self.domain_dicts[l] = {_[0] for _ in vocab if _[1] == l and _[2] == "1"}
    

    def _get_signal(self, row):
        signal_dict = self.signal_dicts[row["label"]]
        tokens = row["tokens"].split(" ")
        signal = ["{:.5f}".format(signal_dict[t])
                  if t in signal_dict else "0.0" for t in tokens]
        return " ".join(signal)

    
    def _get_domain(self, row):
        tokens = row["tokens"].split(" ")
        tokens = [wnl.lemmatize(token, "n") for token in tokens] # lemmatization nouns.
        tokens = [wnl.lemmatize(token, "v") for token in tokens] # lemmatization verbs.
        tokens = [wnl.lemmatize(token, "a") for token in tokens] # lemmatization adjectives.
        domain = ["1" if t in self.domain_dicts["positive"] else
                  ("-1" if t in self.domain_dicts["negative"] else "0")
                  for t in tokens]
        return " ".join(domain)

    
    def signal(self, data_dir):
        df = pd.read_csv(data_dir, sep="\t")
        df["linear_signal"] = df.apply(self._get_signal, axis=1)
        df["domain_knowledge"] = df.apply(self._get_domain, axis=1)
        df.to_csv(data_dir, index=False, sep="\t")


if __name__ == "__main__":
    wnl = WordNetLemmatizer()
    signaler = DataSignaler()
    signaler.signal(signaler.data_dirs[0])
    signaler.signal(signaler.data_dirs[1])
    signaler.signal(signaler.data_dirs[2])
