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

        # Load factchecks (train set).
        factcheck_path = os.path.join(self.data_dir, "train.tsv")
        factcheck = pd.read_csv(factcheck_path, delimiter="\t", names=["label", "tokens"])

        # Stats.
        factcheck["tokens"] = factcheck["tokens"].apply(lambda t: t.split(" "))
        factcheck["len"] = factcheck["tokens"].apply(lambda t: len(t))
        print(factcheck[factcheck["len"] > 2048])
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        bins = [100 * i for i in range(30)]
        factcheck["len"].plot.hist(ax=ax, bins=bins, color="k", lw=0, alpha=0.6)
        plt.savefig(os.path.join("figs", "token_len.pdf"), bbox_inches="tight", pad_inches=0)
        
        # Load word weights from logistic regression.
        word_weight_path = os.path.join(self.data_dir, "linear.analyze", "word_weight.json")
        ww = pd.read_json(word_weight_path, lines=True)
        ww["w"] = ww["false"].abs() + ww["true"].abs() + \
                  ww["mixture"].abs() + ww["unproven"].abs()
        top_w = set(ww.sort_values("w").tail(100)["word"])
        
        # Find word location.
        factcheck["locs"] = factcheck["tokens"].apply(lambda t: [i for i, w in enumerate(t) if w in top_w])
        locs = factcheck["locs"].apply(pd.Series).unstack().reset_index(drop=True).dropna()
        
        # Plot word location.
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        bins = [100 * i for i in range(30)]
        locs.plot.hist(ax=ax, bins=bins, color="k", lw=0, alpha=0.6)
        plt.savefig(os.path.join("figs", "token_loc.pdf"), bbox_inches="tight", pad_inches=0)
        print(locs)


if __name__ == "__main__":
    DataPlotter().plot()
