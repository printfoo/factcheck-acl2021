# coding: utf-8


import os
import numpy as np
import pandas as pd

from runners.metrics import precision, recall, f1, accuracy, percentage
metric_funcs = {"precision": precision, "percentage": percentage}


def _thresholding(r, th):
    return [float(_ > th) for _ in r]


def _numerize(r):
    return [float(_) for _ in r.split(" ")]


def test_threshold(df, thresholds):
    df["true"] = df["rationale_true"].apply(_numerize)
    df["score"] = df["rationale_pred"].apply(_numerize)
    df["mask"] = df["mask"].apply(_numerize)
    
    for th in thresholds:
        df["pred"] = df["score"].apply(lambda r: _thresholding(r, th))

        for metric_name, metric_func in metric_funcs.items():
            df[metric_name] = df.apply(lambda r: metric_func(r["true"], r["pred"], r["mask"], average="binary"), axis=1)
            print(th, "\t", metric_name, "\t", df[metric_name].mean())


def binarize(out_path, args):

    # set_names = ["train", "dev", "test"]  # Analyze all.
    set_names = ["dev", "test"]
    for set_name in set_names:
        print(set_name)
        set_path = os.path.join(out_path, set_name + ".tsv")
        df = pd.read_csv(set_path, sep="\t")
        test_threshold(df, args.test_thresholds)
