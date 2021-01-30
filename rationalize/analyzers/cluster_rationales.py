# coding: utf-8


import os, json, shutil, random, heapq
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from wordcloud import WordCloud
from cairosvg import svg2png
from PIL import Image
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
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 14))
        R = dendrogram(Z, color_threshold=Z[-self.cluster_num + 1,2],
                       above_threshold_color=default_color, ax=ax, orientation="left")
        ax.set_ylim([max(max(R["icoord"]))+100, min(min(R["icoord"]))-100])
        ax.set_yticks([])
        ax.set_xlim([1.1, 0])
        ax.set_xticks([1, 0.5, 0])
        ax.set_xlabel("cosine distance", fontproperties=font_bold)
        for edge in ["right", "left", "top", "bottom"]:
             ax.spines[edge].set_visible(False)
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0)
        return R


    def _plot_wordcloud(self, rationale_freq, rationale_num, cluster_color, fig_path):
        
        # Generate a mask, rounded rectangular on left side.
        mask_svg_path = fig_path.replace(".png", "_mask.svg")
        mask_png_path = fig_path.replace(".png", "_mask.png")
        width = 600
        rounded = width * 0.025
        stroke = 3
        height_rounded = rationale_num  # In scale to the # of leaves.
        height = height_rounded - 2 * rounded - stroke
        
        mask_svg = """
        <svg
            xmlns="http://www.w3.org/2000/svg"
            width="{width_rounded}"
            height="{height_rounded}"
            viewBox="{stroke_half} -{stroke_half} {width_rounded} {height_rounded}">
        <rect width="100%" height="100%" fill="none"/>
        <path d="M0,0
            h{width}
            q{rounded}, 0 {rounded},{rounded}
            v{height}
            q0,{rounded} -{rounded},{rounded}
            h-{width}
            z"
            fill="{color}"
            fill-opacity="0.2"
            stroke="{color}"
            stroke-width="{stroke}">
        </path>
        </svg>
        """.format(
            width=width,
            height=height,
            rounded=rounded,
            width_rounded=width + rounded,
            height_rounded=height_rounded,
            color=cluster_color,
            stroke=stroke,
            stroke_half=stroke/2,
        )
        with open(mask_svg_path, "w") as f:
            f.write(mask_svg)
        svg2png(
            bytestring=mask_svg.replace("none", "#ffffff"),
            write_to=mask_png_path
        )
        mask = np.array(Image.open(mask_png_path))

        rationale_freq = {k: min(v, 30) for k, v in rationale_freq.items()}
        print(rationale_freq)

        # Plot wordcloud.
        wc = WordCloud(mask=mask,
                       background_color=None,  # Transparent background.
                       mode="RGBA",  # Transparent background.
                       color_func=lambda *_, **__: default_color,  # Default color text.
                       relative_scaling=.5,
                       prefer_horizontal=0.9999,
                       max_font_size=50)
        wc.generate_from_frequencies(rationale_freq)
        wc.to_file(fig_path)


    def _get_rationale_freq(self, g):
        rationale_freq = {}
        for word, freq in g[["rationale", "count"]].values:
            rationale_freq[word] = freq
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

        fig_path = os.path.join(label_cluster_path, "dendrogram.svg")
        R = self._plot_dendrogram(Z, fig_path)
        
        words_path = os.path.join(label_cluster_path, "clusters.json")
        words = open(words_path, "w")
        for _, g in df.groupby("cluster"):
            cluster_id = int(g["cluster"].tolist()[-1] - 1)
            cluster_color = regular_colors[cluster_id % len(regular_colors)]
            rationale_num = R["color_list"].count(cluster_color)
            fig_path = os.path.join(label_cluster_path, "{:03d}".format(cluster_id) + ".png")
            rationale_freq = self._get_rationale_freq(g)
            words.write(json.dumps(rationale_freq) + "\n")
            self._plot_wordcloud(rationale_freq, rationale_num, cluster_color, fig_path)
        words.close()


def clust(vector_path, cluster_path, train_args):
    if not os.path.exists(cluster_path):
        os.mkdir(cluster_path)
    cluster = Cluster(vector_path, cluster_path, train_args)
    for label in cluster.labels:
        cluster.clust(label)
