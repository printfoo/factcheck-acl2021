# coding: utf-8


import os, shutil, multidict, random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from wordcloud import WordCloud
np.random.seed(0)
random.seed(0)


# Matplotlib settings.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 14
font_bold = FontProperties()
font_bold.set_weight("bold")


# Color settings.
"""
[
    "#%02x%02x%02x" % (int(round(_[0]*255)), int(round(_[1]*255)), int(round(_[2]*255)))
    for _ in sns.color_palette("pastel", n_colors=10)
]
"""
colors_num = 10
regular_colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
pastel_colors = [
    "#a1c9f4", "#ffb482", "#8de5a1", "#ff9f9b", "#d0bbff",
    "#debb9b", "#fab0e4", "#cfcfcf", "#fffea3", "#b9f2f0"
]
default_color = "#000000"


class Cluster(object):
    """
    Dataset Cluster for movie reviews.
    """


    def __init__(self, vector_path, cluster_path, train_args):        
        self.embeddings_path = os.path.join(
            vector_path,
            "rationale_embeddings" + train_args.cluster_postfix + ".csv"
        )
        
        self.df = pd.read_csv(self.embeddings_path).dropna()
        self.df["embeddings"] = self.df["embeddings"].apply(
            lambda e: [float(_) for _ in e.split(" ")]
        )
        print("Number of rationales:", len(self.df))
        
        self.labels = train_args.cluster_labels
        self.cluster_path = cluster_path
        self.cluster_num = train_args.cluster_num


    def _plot_dendrogram(self, Z, fig_path):
        hierarchy.set_link_color_palette(regular_colors)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 15))
        R = dendrogram(Z, color_threshold=Z[-self.cluster_num + 1,2],
                       above_threshold_color=default_color, ax=ax, orientation="left")
        ax.set_ylim([max(max(R["icoord"]))+100, min(min(R["icoord"]))-100])
        ax.set_yticks([])
        ax.set_xlim([1.1, 0])
        ax.set_xlabel("Cosine distance of embeddings", fontproperties=font_bold)
        for edge in ["right", "left", "top", "bottom"]:
             ax.spines[edge].set_visible(False)
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
        return R


    def _plot_wordcloud(self, rationale_freq, rationale_num, cluster_color, fig_path):
        wc = WordCloud(width=2000,
                       height=rationale_num * 5,  # In scale to the # of leaves.
                       background_color=None,  # Transparent background.
                       mode="RGBA",  # Transparent background.
                       color_func=lambda *_, **__: default_color,  # Default color text.
                       relative_scaling=0.,
                       prefer_horizontal=0.999)
        wc.generate_from_frequencies(rationale_freq)
        wc.to_file(fig_path)


    def _get_rationale_freq(self, g):
        rationale_freq = multidict.MultiDict()
        for word, freq in g[["rationale", "count"]].values:
            rationale_freq.add(word, freq)
        return rationale_freq

    
    def clust(self, label):
        label_cluster_path = os.path.join(self.cluster_path, label)
        if not os.path.exists(label_cluster_path):
            os.mkdir(label_cluster_path)
            
        df = self.df[self.df["label"] == label].reset_index()
        X = np.array(df["embeddings"].tolist())
        
        Z = linkage(X, method="complete", metric="cosine", optimal_ordering=False)  # Linkage metrix.
        T = fcluster(Z, criterion="maxclust", t=self.cluster_num)  # Cluster labels.
        df["cluster"] = pd.Series(T)

        fig_path = os.path.join(label_cluster_path, "dendrogram.png")
        R = self._plot_dendrogram(Z, fig_path)
        
        for _, g in df.groupby("cluster"):
            cluster_id = int(g["cluster"].tolist()[-1] - 1)
            cluster_color = regular_colors[cluster_id % len(regular_colors)]
            rationale_num = R["color_list"].count(cluster_color)
            fig_path = os.path.join(label_cluster_path, "{:03d}".format(cluster_id) + ".png")
            rationale_freq = self._get_rationale_freq(g)
            self._plot_wordcloud(rationale_freq, rationale_num, cluster_color, fig_path)


def clust(vector_path, cluster_path, train_args):
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)
    cluster = Cluster(vector_path, cluster_path, train_args)
    for label in cluster.labels:
        cluster.clust(label)
