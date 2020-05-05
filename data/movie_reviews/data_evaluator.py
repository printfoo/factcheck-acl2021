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
    row["%"] = sum(row[pred_col]) / len(row[pred_col])
    return row


def get_num(r):
    r_shifted = r[1:] + r[-1:]
    num = [abs(r1 - r2) for r1, r2 in zip(r, r_shifted)]
    return sum(num) / 2


class DataEvaluator(object):
    """
    Dataset evaluator for personal attacks of linear models.
    Evaluate dev.tsv and test.tsv.
    """

    def __init__(self):
        self.data_dirs = ["train.tsv", "dev.tsv", "test.tsv"]

    
    def evaluate(self, data_dir):
        df = pd.read_csv(data_dir, sep="\t")
        df["linear_signal"] = df["linear_signal"].apply(lambda s: [int(float(_) >= 0.0711) for _ in s.split()])
        df["domain_knowledge"] = df["domain_knowledge"].apply(lambda d: [int(_) for _ in d.split()])
        df["domain_knowledge_abs"] = df["domain_knowledge"].apply(lambda d: [abs(_) for _ in d])
        df["rationale_annotation"] = df["rationale_annotation"].apply(lambda r: [int(_) for _ in r.split()])

        rationale_len = df["rationale_annotation"].apply(sum).mean()
        rationale_num = df["rationale_annotation"].apply(get_num).mean()
        print("rationale_len:", rationale_len)
        print("rationale_num:", rationale_num)
        print()
        
        print("Rationale evaluation for:", data_dir)
        for pred_col in ["linear_signal", "domain_knowledge_abs"]:
            df = df.apply(lambda row: get_metrics(row, pred_col), axis=1)
            print(pred_col)
            print(df.mean())
        print()

        print("Prediction evaluation for:", data_dir)
        print("linear_signal\n[results shown in the main model.]")
        df["true"] = df["label"].apply(lambda l: 1 if l == "positive" else 0)
        df["d_pred"] = df.apply(lambda r: 1 if sum(r["domain_knowledge"]) > 0 else 0, axis=1)
        print("domain_knowledge")
        print("a\t", accuracy_score(df["d_pred"], df["true"]))
        print("p\t", precision_score(df["d_pred"], df["true"], average="macro"))
        print("r\t", recall_score(df["d_pred"], df["true"], average="macro"))
        print("f\t", f1_score(df["d_pred"], df["true"], average="macro"))
        print()


if __name__ == "__main__":
    evaluator = DataEvaluator()
    evaluator.evaluate(evaluator.data_dirs[1])
    evaluator.evaluate(evaluator.data_dirs[2])
