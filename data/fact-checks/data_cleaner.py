# coding: utf-8


import os, json, random, nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
random.seed(0)


misinfo = {"unproven", "unconfirmed", "undetermined", "in progress", "probably", "legend", "maybe",
           "mix", "incomplete", "partly", "sort of", "not quite",
           "false", "hoax", "scam", "fraud", "fiction", "satire", "incorrect", "inaccurate", 
           "miscaptioned", "misattributed", "outdated", "not any", "no longer", "was true", "true but"}
info = {"true", "real", "correct", "accurate"}


def process_verdict(verdict):
    for label, words in [("misinfo", misinfo), ("info", info)]:
        for word in words:
            if word in verdict:
                return label
    return np.nan
    

def process_tokens(c):
    tokens = tokenizer.tokenize(c)
    tokens = ["<" + t[:-5] + ">" if t.endswith("TOKEN") else t.lower() for t in tokens]
    if len(tokens) > 1000:
        tokens = tokens[:500] + ["<MORE>"] + tokens[-500:]
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
        
        for lim in [512, 1024, 2048, 4096]:
            percentage = len(factcheck[factcheck["len"] <= lim]) / len(factcheck)
            print(lim, "\t", percentage)
        
        # Other information.
        factcheck["rationale_annotation"] = " "
        factcheck["linear_signal"] = " "
        factcheck["domain_knowledge"] = " "
        
        # Split and save.
        selected_cols = ["label", "tokens", "rationale_annotation", "linear_signal", "domain_knowledge"]
        train = factcheck.sample(frac=0.8)
        train[selected_cols].to_csv("train.tsv", sep="\t", index=False)
        factcheck = factcheck.drop(train.index)
        dev = factcheck.sample(frac=0.5)
        dev[selected_cols].to_csv("dev.tsv", sep="\t", index=False)
        test = factcheck.drop(dev.index)
        test[selected_cols].to_csv("test.tsv", sep="\t", index=False)
        

if __name__ == "__main__":
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    DataCleaner().clean()
