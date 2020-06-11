# coding: utf-8


import os, shutil, multidict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from wordcloud import WordCloud


class Cluster(object):
    """
    Dataset Cluster for movie reviews.
    """


    def __init__(self, label):
        self.label = label
        self.cluster_dir = "rationale_clusters"
        self.embeddings_path = "rationale_embeddings.csv"
        self.df = pd.read_csv(self.embeddings_path)
        self.df = self.df[self.df["label"] == self.label].dropna().reset_index()
        self.df["embeddings"] = self.df["embeddings"].apply(lambda e: [float(_) for _ in e.split(" ")])
        self.X = np.array(self.df["embeddings"].tolist())
        self.model = AgglomerativeClustering(affinity="cosine", linkage="average",
                                             n_clusters=None, distance_threshold=0.8)
        self.color = {"positive": "seagreen", "negative": "indianred"}[self.label]

    
    def _plot_dendrogram(self, fig_path, **kwargs):
        counts = np.zeros(self.model.children_.shape[0])  # Count samples under each node.
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # Leaf node.
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts]).astype(float)
        print(linkage_matrix)
        exit()

        dendrogram(linkage_matrix, **kwargs)
        plt.xlabel("Number of points in node (or index of point if no parenthesis).")
        plt.savefig(fig_path)


    def _plot_wordcloud(self, rationale_freq, rationle_color, fig_path):
        wc = WordCloud(background_color="white", color_func=rationle_color,
                       width=400, height=300, relative_scaling=0.5)
        wc.generate_from_frequencies(rationale_freq)
        wc.to_file(fig_path)


    def _get_rationale_freq(self, g):
        rationale_freq = multidict.MultiDict()
        for word, freq in g[["rationale", "count"]].values:
            rationale_freq.add(word, freq)
        return rationale_freq

    
    def clust(self):
        self.model.fit(self.X)
        clusters = self.model.labels_
        print(self.label, "# of clusters:", max(clusters))
        self.df["cluster"] = pd.Series(clusters)
        
        fig_path = os.path.join(self.cluster_dir, self.label + "-rationale-dendrogram.png")
        self._plot_dendrogram(fig_path, truncate_mode="level", p=5)
        exit()
        for _, g in self.df.groupby("cluster"):
            if len(g) < 10:  # Do not plot very small clusters.
                continue
            top_word = g.sort_values("count")["rationale"].tolist()[-1]
            cluster_id = "{:03d}".format(g["cluster"].tolist()[-1])
            fig_path = os.path.join(self.cluster_dir, self.label + "-" + cluster_id + "-" + top_word + ".png")
            rationale_freq = self._get_rationale_freq(g)
            rationle_color = lambda *args, **kwargs: self.color
            self._plot_wordcloud(rationale_freq, rationle_color, fig_path)


if __name__ == "__main__":
    if os.path.exists("rationale_clusters"):
        shutil.rmtree("rationale_clusters")
    os.mkdir("rationale_clusters")
    for label in ["positive", "negative"]:
        cluster = Cluster(label)
        cluster.clust()
