# coding: utf-8


import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(row):
    row["a"] = accuracy_score(row["rationale_annotation"], row["linear_signal"])
    if sum(row["linear_signal"]) >= 1:
        row["p"] = precision_score(row["rationale_annotation"], row["linear_signal"], average="binary")
    else:
        row["p"] = np.nan
    if sum(row["rationale_annotation"]) >= 1:
        row["r"] = recall_score(row["rationale_annotation"], row["linear_signal"], average="binary")
    else:
        row["r"] = np.nan
    if sum(row["rationale_annotation"]) >= 1 and sum(row["linear_signal"]) >= 1:
        row["f"] = f1_score(row["rationale_annotation"], row["linear_signal"], average="binary")
    else:
        row["f"] = np.nan
    return row


class DataEvaluator(object):
    """
    Dataset evaluator for personal attacks of linear models.
    Evaluate dev.tsv and test.tsv.
    """

    def __init__(self):
        self.data_dirs = ["train.tsv", "dev.tsv", "test.tsv"]

    
    def evaluate(self, data_dir):
        df = pd.read_csv(data_dir, sep="\t")
        df["linear_signal"] = df["linear_signal"].apply(lambda s: [int(float(_) >= 0.25) for _ in s.split()])
        df["rationale_annotation"] = df["rationale_annotation"].apply(lambda r: [int(_) for _ in r.split()])
        df = df.apply(get_metrics, axis=1)
        print(df.mean())


if __name__ == "__main__":
    evaluator = DataEvaluator()
    evaluator.evaluate(evaluator.data_dirs[1])
    evaluator.evaluate(evaluator.data_dirs[2])
