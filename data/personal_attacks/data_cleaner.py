# coding: utf-8


import os, json
import pandas as pd
import numpy as np
import nltk


def process_comment(row):

    # This two lines can NOT be changed as comments align with rationales.
    tokens = row["comment"].replace("TAB_TOKEN", "\t").replace("NEWLINE_TOKEN", "\n").lower().strip()
    tokens = tokenizer.tokenize(tokens)

    # Invalid.
    if len(tokens) < 2:
            return np.nan

    # Remove leading "`,:".
    while "`" in tokens[0] or "," in tokens[0] or ":" in tokens[0]:
        if len(tokens) < 2:  # Invalid.
            return np.nan
        tokens = tokens[1:]
        if row["rationale"] == row["rationale"]:  # If not NaN.
            row["rationale"] = row["rationale"][1:]

    # Remove trailing "`".
    if tokens[-1] == "`":
        tokens = tokens[:-1]
        if row["rationale"] == row["rationale"]:  # If not NaN.
            row["rationale"] = row["rationale"][:-1]
    
    row["comment"] = " ".join(tokens)

    # Validity checks.
    if row["rationale"] == row["rationale"]:  # If not NaN.
        assert len(row["comment"].split(" ")) == len(row["rationale"])
    if row["split_y"] == row["split_y"]:  # If not NaN.
        assert row["split_x"] == row["split_y"]

    return row


class DataCleaner(object):
    """
    Dataset cleaner for personal attacks.
    """

    def __init__(self, data_dir="raw", score_threshold=0.3):
        """
        Inputs:
            data_dir -- the directory of the dataset.
            score_threshold -- the threshold (0-1) for pos/neg.
        """
        self.data_dir = data_dir
        self.score_threshold = score_threshold
        self.label_vocab = {}
        

    def clean(self):

        # Load sentence.
        sentence_path = os.path.join(self.data_dir, "attack_annotated_comments.tsv")
        sentence = pd.read_csv(sentence_path, delimiter="\t")

        # Load label.
        label_path = os.path.join(self.data_dir, "attack_annotations.tsv")
        label = pd.read_csv(label_path, delimiter="\t")
        label = label.groupby("rev_id").mean()

        # Load rationale.
        rationale_dev_path = os.path.join(self.data_dir, "wiki_attack_dev_rationale.csv")
        rationale_dev = pd.read_csv(rationale_dev_path)
        rationale_dev["split"] = "dev"
        rationale_test_path = os.path.join(self.data_dir, "wiki_attack_test_rationale.csv")
        rationale_test = pd.read_csv(rationale_test_path)
        rationale_test["split"] = "test"
        rationale = pd.concat([rationale_dev, rationale_test])
        rationale["rev_id"] = rationale["platform_comment_id"]
        rationale["rationale"] = rationale["rationale"].apply(lambda r: "".join(str(_) for _ in eval(r)))
        
        # Merge sentence and label.
        df = sentence.merge(label, on="rev_id", how="inner").merge(rationale, on="rev_id", how="left")
        df = df.apply(process_comment, axis=1)
        df = df.dropna(subset={"comment"})
        df["split_y"] = df["split_y"].fillna("train")
        df["label"] = df["attack"].apply(lambda a: "attack" if a >= self.score_threshold else "not_attack")
        df = df.fillna("")

        # Save data.
        for split in ["train", "dev", "test"]:
            data = df[df["split_y"] == split][["label", "comment", "rationale"]]
            data.to_csv(split + ".tsv", index=False, sep="\t")


if __name__ == "__main__":
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    DataCleaner().clean()
