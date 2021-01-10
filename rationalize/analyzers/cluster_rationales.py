# coding: utf-8


import os, shutil, multidict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from wordcloud import WordCloud


class Cluster(object):
    """
    Dataset Cluster for movie reviews.
    """


    def __init__(self, vector_path, cluster_path):        
        self.embeddings_path = os.path.join(vector_path, "rationale_embeddings.csv")
        
        self.df = pd.read_csv(self.embeddings_path).dropna()
        self.df = self.df[self.df["count"] > 1]
        self.df["embeddings"] = self.df["embeddings"].apply(lambda e: [float(_) for _ in e.split(" ")])
        print("Number of rationales:", len(self.df))
        
        self.labels = set(self.df["label"])
            
        self.cluster_path = cluster_path
        self.cluster_num = 10
        self.colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", 
                       "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    
    def _plot_dendrogram(self, Z, fig_path):
        hierarchy.set_link_color_palette(self.colors)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 15))
        R = dendrogram(Z, color_threshold=Z[-self.cluster_num + 1,2],
                       above_threshold_color="#555555", ax=ax, orientation="left")
        ax.set_ylim([max(max(R["icoord"]))+10, min(min(R["icoord"]))-10])
        plt.axis("off")
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
        return R


    def _plot_wordcloud(self, rationale_freq, rationale_num, cluster_color, fig_path):
        wc = WordCloud(width=1000, height=rationale_num * 5,
                       background_color="white",
                       color_func=lambda *_, **__: cluster_color, 
                       relative_scaling=0., prefer_horizontal=0.99)
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
            cluster_color = self.colors[cluster_id]
            rationale_num = R["color_list"].count(cluster_color)
            assert len(g) - 1 == rationale_num, "Mismatch!"
            fig_path = os.path.join(label_cluster_path, "{:03d}".format(cluster_id) + ".png")
            rationale_freq = self._get_rationale_freq(g)
            self._plot_wordcloud(rationale_freq, rationale_num, cluster_color, fig_path)


def clust(vector_path, cluster_path):
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)
    cluster = Cluster(vector_path, cluster_path)
    for label in cluster.labels:
        cluster.clust(label)
