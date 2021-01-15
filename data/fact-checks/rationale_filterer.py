# coding: utf-8


import os, json
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine


rationale_path = os.path.join("soft_rationalizer_w_domain.vector", "rationale_embeddings.csv")

misinfo = {
    # Snopes verdicts.
    "undetermined", "maybe", "incomplete", "outdate", "legend", "ficti", "satir",
    "fals", "hoax", "scam", "incorrect", "inaccurate", "miscaption", "misattribut",
    # Fake news. It’s complicated.
    "fabricat", "manipulat", "imposter", "mislead", "misled", "parody",
    # Added.
    "conspirac", "jok", "prank", "spoof", "doctored", "mistak", "dubious", "exaggerat", 
    "unfound", "baseless", "vague", "propaganda", "phony", "humor", "bogus", "gossip",
    "misrepresent", "error", "spurious", "flaw", "unsubstantiat", "apocryph", "discard", "bias",
}
info = {
    # Snopes verdicts.
    "true", "real", "correct", "accurate",
}
labels_dict = {"misinfo": misinfo, "info": info}
excludes = {
    "report", "despite", "one", "exist", "clear", "share", "falsi", "unrelated", "neon", "old", 
    "sources", "business", "mixture", "hearsay", "breathless", "limerick", "authentic", "coupon", 
    "content", "numerous", "web", "site", "website", "com", "account", "facebook", "reddit", 
    "blog", "post", "news", "publi", "push",
}


class RationaleFilter(object):
    """
    Rationale filter for fact-checks.
    """

    def __init__(self, rationale_path):
        self.df= pd.read_csv(rationale_path)
        self.df = self.df.dropna()
        self.df["embeddings"] = self.df["embeddings"].apply(
            lambda e: [float(_) for _ in e.split(" ")]
        )
        self.df["rlen"] = self.df["rationale"].apply(
            lambda r: len(r.split(" "))
        )
        self.df = self.df[self.df["rlen"] <= 5]
        self.misinfo = misinfo
        self.misinfo_embeddings = {}
        self.s = 0.3  # Similarity threshold.
        
    
    def _get_misinfo_embeddings(self, r):
        for word in self.misinfo:
            if word in r["rationale"]:
                self.misinfo_embeddings[word] = r["embeddings"]
                self.misinfo -= {word}
                break

    
    def _filter_row(self, r):
        r["contain"] = False
        r["similar"] = False
        for word, word_e in self.misinfo_embeddings.items():
            if word in r["rationale"]:
                r["contain"] = True
                r["embeddings"] = [0.3*x+0.7*y for x, y in zip(r["embeddings"], word_e)]
                break
            if cosine(r["embeddings"], word_e) < self.s:
                r["similar"] = True
                break
        r["keep"] = r["contain"] or r["similar"]
        for word in excludes:
            if word in r["rationale"]:
                r["keep"] = False
                break
        return r
    
    
    def rfilter(self):
        self.df.apply(self._get_misinfo_embeddings, axis=1)
        print("Root words:", len(self.misinfo_embeddings))
        self.df = self.df.apply(self._filter_row, axis=1)
        self.filtered = self.df[self.df["keep"]]
        self.filtered["embeddings"] = self.filtered["embeddings"].apply(
            lambda e: " ".join([str(_) for _ in e])
        )
        print(self.filtered)
        self.filtered.to_csv(
            rationale_path.replace(".csv", "_filtered.csv"),
            index=False
        )
        

if __name__ == "__main__":
    rfilter = RationaleFilter(rationale_path)
    rfilter.rfilter()
