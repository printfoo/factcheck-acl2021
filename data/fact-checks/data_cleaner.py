# coding: utf-8


import os, json, random, nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
random.seed(0)


misinfo = {
    # Snopes verdicts.
    "unproven", "unconfirmed", "undetermined", "probably", "maybe",
    "mixture", "incomplete", "partly", "outdate",
    "legend", "ficti", "satir", "superstition",  
    "fals", "hoax", "scam", "fraud",
    "incorrect", "inaccurate", "miscaption", "misattribut",
    # For verdict only.
    "sort of", "not quite", "not likely", "in progress", 
    "not any", "no longer", "was true", "true but",
    # Fake news. It’s complicated.
    "fabricat", "manipulat", "imposter", "mislead", "misled", "parody",
    # Added.
    "conspirac", "jok", "prank", "spoof", "doctored", "mistak", "plot",
    "dubious", "exaggerat", "myth", "unfound", "fool", "baseless", "vague",
    "deliberat", "unrelat", "propaganda", "phony", "humor", "bogus", "gossip",
    "misrepresent", "error", "spurious", "flaw", "unsubstantiat", "apocryph",
    "unverif", "dismiss", "nonsens", "contradict", "unsupport", "discard", 
    "bias", "conject", "innuendo", "nonexist", "disreput", "intentional",
}
info = {
    # Snopes verdicts.
    "true", "real", "correct", "accurate",
}
labels_dict = {"misinfo": misinfo, "info": info}
masks = {
    "false", "true", "claim", "stat", "quot",
    "origin", "story", "article", "rumor", "evidence", "proof"
}


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
    for i, t in enumerate(tokens):  # Mask too signaling words.
        for word in masks:
            if word in t:
                tokens[i] = "<MASK>"
    return pd.Series([" ".join(tokens), len(tokens)])


def process_domain(r):
    tokens = r["tokens"].split(" ")
    domain = ["0" for t in tokens]
    for i, t in enumerate(tokens):
        for word in labels_dict[r["label"]]:
            if word in t:
                domain[i] = "1"
    return " ".join(domain)


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

        # _ = factcheck.groupby("verdict").count()["url"].sort_values(ascending=False).index.tolist()
        
        # Process verdict.
        factcheck["label"] = factcheck["verdict"].apply(process_verdict)
        factcheck = factcheck.dropna()
        
        # Process tokens.
        factcheck[["tokens", "len"]] = factcheck["content"].apply(process_tokens)
        factcheck = factcheck.dropna()
        
        # Print symbols.
        tokens = set(" ".join(factcheck["tokens"].tolist()).split(" "))
        symbols = {t for t in tokens if t.startswith("<") and t.endswith(">")}
        print(symbols)
        
        for lim in [512, 1024, 2048, 4096]:
            percentage = len(factcheck[factcheck["len"] <= lim]) / len(factcheck)
            print(lim, "\t", percentage)
            
        # Process domain knowledge.
        factcheck["domain_knowledge"] = factcheck.apply(process_domain, axis=1)
        
        # Other information.
        factcheck["rationale_annotation"] = " "
        factcheck["linear_signal"] = " "
        
        # Split and save.
        selected_cols = ["label", "tokens", "rationale_annotation", 
                         "linear_signal", "domain_knowledge", "date"]
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
