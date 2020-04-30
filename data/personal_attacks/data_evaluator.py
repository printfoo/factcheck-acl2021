# coding: utf-8


import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def get_metrics(row, pred_col, true_col="rationale_annotation", average="binary"):
    row["a"] = accuracy_score(row[true_col], row[pred_col])
    if sum(row[pred_col]) >= 1:
        row["p"] = precision_score(row[true_col], row[pred_col], average=average)
    else:
        row["p"] = np.nan
    if sum(row[true_col]) >= 1:
        row["r"] = recall_score(row[true_col], row[pred_col], average=average)
    else:
        row["r"] = np.nan
    if sum(row[true_col]) >= 1 and sum(row[pred_col]) >= 1:
        row["f"] = f1_score(row[true_col], row[pred_col], average=average)
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
        df["linear_signal"] = df["linear_signal"].apply(lambda s: [int(float(_) >= 0.251) for _ in s.split()])
        df["domain_knowledge"] = df["domain_knowledge"].apply(lambda r: [int(_) for _ in r.split()])
        df["rationale_annotation"] = df["rationale_annotation"].apply(lambda r: [int(_) for _ in r.split()])
        print("Rationale evaluation for:", data_dir)
        for pred_col in ["linear_signal", "domain_knowledge"]:
            df = df.apply(lambda row: get_metrics(row, pred_col), axis=1)
            print(pred_col)
            print(df.mean())
        print()
        
        print("Prediction evaluation for:", data_dir)
        print("linear_signal\n[results shown in the main model.]")
        df["d_pred"] = df["domain_knowledge"].apply(lambda d: 1 if sum(d) > 0 else 0)
        df["true"] = df["label"].apply(lambda l: 1 if l == "attack" else 0)
        print("domain_knowledge")
        print("a\t", accuracy_score(df["d_pred"], df["true"]))
        print("p\t", precision_score(df["d_pred"], df["true"], average="macro"))
        print("r\t", recall_score(df["d_pred"], df["true"], average="macro"))
        print("f\t", f1_score(df["d_pred"], df["true"], average="macro"))


if __name__ == "__main__":
    evaluator = DataEvaluator()
    evaluator.evaluate(evaluator.data_dirs[1])
    evaluator.evaluate(evaluator.data_dirs[2])
