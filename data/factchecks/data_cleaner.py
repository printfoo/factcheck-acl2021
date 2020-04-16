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
    Dataset cleaner for fact-checks.
    """

    def __init__(self, data_dir="raw"):
        """
        Inputs:
            data_dir -- the directory of the dataset.
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
        factcheck["comment"] = factcheck["tokens"]
        factcheck["rationale"] = ""
        
        # Split and save.
        train = factcheck.sample(frac=0.8)
        train[["label", "comment", "rationale"]].to_csv("train.tsv", sep="\t", index=False)
        factcheck = factcheck.drop(train.index)
        dev = factcheck.sample(frac=0.5)
        dev[["label", "comment", "rationale"]].to_csv("dev.tsv", sep="\t", index=False)
        test = factcheck.drop(dev.index)
        test[["label", "comment", "rationale"]].to_csv("test.tsv", sep="\t", index=False)
        

if __name__ == "__main__":
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    DataCleaner().clean()
