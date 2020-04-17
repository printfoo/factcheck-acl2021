# coding: utf-8


import os, json
import pandas as pd


class DataSignaler(object):
    """
    Dataset signaler for personal attacks.
    Add signals to train.tsv.
    """

    def __init__(self):
        self.train_dir = "train.tsv"
        self.vocab_dir = "vocab.json"
        self.dist_dir = "train_with_word_freq.tsv"
        

    def count_words(self):
        df = pd.read_csv(self.train_dir, sep="\t")
        wc = {}
        for w in " ".join(df["comment"].values).split(" "):
            if w not in wc:
                wc[w] = 0
            wc[w] += 1
        print(len(wc))
        wc = {w: c for w, c in wc.items() if c >= 10000}
        with open(self.vocab_dir, "w") as f:
            f.write(json.dumps(wc))
            
    
    def add_word_dist(self):
        df = pd.read_csv(self.train_dir, sep="\t")
        with open(self.vocab_dir, "r") as f:
            wc = json.loads(f.read())
        
        def word_dist(row):
            comment = row["comment"].split(" ")
            for w in wc:
                row[w] = comment.count(w)
            return row
        
        df = df.apply(word_dist, axis=1)
        df.to_csv(self.dist_dir, sep="\t")
            

if __name__ == "__main__":
    datasignaler = DataSignaler()
    if not os.path.exists(datasignaler.train_dir):
        print("Please run data_cleaner.py first.")
    elif not os.path.exists(datasignaler.vocab_dir):
        DataSignaler().count_words()
    else:
        DataSignaler().add_word_dist()
    
