# coding: utf-8


import os, json
import pandas as pd
import numpy as np


class DataCleaner(object):
    """
    Dataset cleaner for movie reviews.
    """

    def __init__(self):
        self.data_dir = os.path.join("raw", "movies")
        self.splits = {"train.tsv": "train.jsonl",
                       "dev.tsv": "val.jsonl",
                       "test.tsv": "test.jsonl"}
        self.label_map = {"NEG": "negative", "POS": "positive"}

                       
    def _process_reviews(self, row):
        tokens_path = os.path.join(self.data_dir, "docs", row["annotation_id"])
        with open(tokens_path, "r") as f:
            tokens = f.read().split()
        rationale = ["0"] * len(tokens)
        for evidences in row["evidences"]:
            for evidence in evidences:
                start = evidence["start_token"]
                end = evidence["end_token"]
                assert tokens[start: end] == evidence["text"].split()
                rationale[start: end] = ["1"] * (end - start)

        row["label"] = self.label_map[row["classification"]]
        row["tokens"] = " ".join(tokens)
        row["rationale_annotation"] = " ".join(rationale)
        row["linear_signal"] = " "
        row["domain_knowledge"] = " "
        return row


    def clean(self, split):
    
        # Load data.
        df = pd.read_json(os.path.join(self.data_dir, self.splits[split]), lines=True)
        
        # Process data.
        df = df.apply(self._process_reviews, axis=1)
        
        # Save data.
        selected_cols = ["label", "tokens", "rationale_annotation", "linear_signal", "domain_knowledge"]
        df[selected_cols].to_csv(split, index=False, sep="\t")


if __name__ == "__main__":
    cleaner = DataCleaner()
    for split in cleaner.splits:
        cleaner.clean(split)
