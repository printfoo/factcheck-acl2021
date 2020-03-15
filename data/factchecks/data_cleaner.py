# coding: utf-8


import os, json, random, nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
random.seed(0)


unproven = {"unproven", "unconfirmed", "undetermined",
    "in progress", "probably", "legend", "maybe"}
mixture = {"mix", "incomplete", "partly", "sort of", "not quite"}
false = {"false", "hoax", "scam", "fraud", "fiction", "satire",
    "incorrect", "inaccurate", "miscaptioned", "misattributed",
    "outdated", "not any", "no longer", "was true", "true but"}
true = {"true", "real", "correct", "accurate"}


def process_verdict(v):
    for label, map in [("unproven", unproven), ("mixture", mixture),
                       ("false", false), ("true", true)]:
        for _ in map:
            if _ in v:
                return label
    return np.nan
    

def process_tokens(c):
    tokens = tokenizer.tokenize(c)
    tokens = ["<" + t[:-5] + ">" if t.endswith("TOKEN") else t.lower() for t in tokens]
    return pd.Series([" ".join(tokens), len(tokens)])


class DataCleaner(object):
    """
    Dataset cleaner for beer reviews.
    """

    def __init__(self, data_dir="raw"):
        """
        Inputs:
            data_dir -- the directory of the dataset.
            score_threshold -- the threshold (0-1) for pos/neg.
        """
        self.data_dir = data_dir
        

    def clean(self):

        # Load factchecks.
        factcheck_path = os.path.join(self.data_dir, "snopes.tsv")
        factcheck = pd.read_csv(factcheck_path, delimiter="\t")
        factcheck = factcheck.drop_duplicates()

        # Process verdict.
        factcheck["label"] = factcheck["verdict"].apply(process_verdict)
        factcheck = factcheck.dropna()
        
        # Process tokens.
        factcheck[["tokens", "len"]] = factcheck["content"].apply(process_tokens)
        factcheck = factcheck.dropna()

        # Stats.
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        bins = [100 * i for i in range(30)]
        factcheck["len"].plot.hist(ax=ax, bins=bins, color="k", lw=0, alpha=0.6)
        plt.savefig(os.path.join("stats", "token_len.pdf"), bbox_inches="tight", pad_inches=0)
        
        # Split and save.
        train = factcheck.sample(frac=0.8)
        train[["label", "tokens"]].to_csv("train.tsv", sep="\t", index=False, header=False)
        factcheck = factcheck.drop(train.index)
        dev = factcheck.sample(frac=0.5)
        dev[["label", "tokens"]].to_csv("dev.tsv", sep="\t", index=False, header=False)
        test = factcheck.drop(dev.index)
        test[["label", "tokens"]].to_csv("test.tsv", sep="\t", index=False, header=False)
        

if __name__ == "__main__":
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    DataCleaner().clean()
