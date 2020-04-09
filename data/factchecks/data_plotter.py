# coding: utf-8


import os, json
import pandas as pd
import matplotlib.pyplot as plt


class DataPlotter(object):
    """
    Dataset plotter for factchecks.
    """

    def __init__(self, data_dir=""):
        """
        Inputs:
            data_dir -- the directory of the dataset.
        """
        self.data_dir = data_dir
        

    def plot(self):

        """
        # Load factchecks (train set).
        factcheck_path = os.path.join(self.data_dir, "train.tsv")
        factcheck = pd.read_csv(factcheck_path, delimiter="\t", names=["label", "tokens"])

        # Stats.
        factcheck["len"] = factcheck["tokens"].apply(lambda t: len(t.split(" ")))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        bins = [100 * i for i in range(30)]
        factcheck["len"].plot.hist(ax=ax, bins=bins, color="k", lw=0, alpha=0.6)
        plt.savefig(os.path.join("figs", "token_len.pdf"), bbox_inches="tight", pad_inches=0)
        """
        
        # Load word weights from logistic regression.
        word_weight_path = os.path.join(self.data_dir, "linear.analyze", "word_weight.json")
        with open(word_weight_path, "r") as f:
            word_weight = json.loads(f.read())
        for label in word_weight:
            pos = [(k, v) for k, v in word_weight[label].items() if v > 0]
            
            print(label, len(pos))
        

if __name__ == "__main__":
    DataPlotter().plot()
