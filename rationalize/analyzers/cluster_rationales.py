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


    def __init__(self, label):
        self.label = label
        
        self.embeddings_path = "rationale_embeddings.csv"
        self.df = pd.read_csv(self.embeddings_path)
        self.df = self.df[self.df["label"] == self.label].dropna().reset_index()
        self.df["embeddings"] = self.df["embeddings"].apply(lambda e: [float(_) for _ in e.split(" ")])
        self.X = np.array(self.df["embeddings"].tolist())
        
        self.cluster_dir = os.path.join("rationale_clusters", self.label)
        if not os.path.exists(self.cluster_dir):
            os.mkdir(self.cluster_dir)
        self.cluster_num = 10
        self.colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", 
                       "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]

    
    def _plot_dendrogram(self, fig_path):
        hierarchy.set_link_color_palette(self.colors)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 15))
        self.R = dendrogram(self.Z, color_threshold=self.Z[-self.cluster_num+1,2],
                            above_threshold_color="#555555", ax=ax, orientation="left")
        ax.set_ylim([max(max(self.R["icoord"]))+10, min(min(self.R["icoord"]))-10])
        plt.axis("off")
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)


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

    
    def clust(self):
        self.Z = linkage(self.X, method="complete", metric="cosine", 
                         optimal_ordering=False)  # Linkage metrix.
    
        self.T = fcluster(self.Z, criterion="maxclust",
                          t=self.cluster_num)  # Cluster labels.
        self.df["cluster"] = pd.Series(self.T)

        fig_path = os.path.join(self.cluster_dir, "dendrogram.png")
        self._plot_dendrogram(fig_path)

        for _, g in self.df.groupby("cluster"):
            cluster_id = g["cluster"].tolist()[-1] - 1
            cluster_color = self.colors[cluster_id]
            rationale_num = self.R["color_list"].count(cluster_color)
            assert len(g) - 1 == rationale_num, "Mismatch!"
            fig_path = os.path.join(self.cluster_dir, "{:03d}".format(cluster_id) + ".png")
            rationale_freq = self._get_rationale_freq(g)
            self._plot_wordcloud(rationale_freq, rationale_num, cluster_color, fig_path)


if __name__ == "__main__":
    if not os.path.exists("rationale_clusters"):
        os.mkdir("rationale_clusters")
    for label in ["positive", "negative"]:
        cluster = Cluster(label)
        cluster.clust()
