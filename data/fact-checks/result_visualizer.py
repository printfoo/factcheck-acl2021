# coding: utf-8


import os, json, math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from statsmodels.stats.proportion import proportion_confint as pc


# Matplotlib settings.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 14
font_bold = FontProperties()
font_bold.set_weight(550)


# Plot style settings.
clusters = [
    "legend, tale, etc.",
    "altered, digitally, etc.",
    "hoax, joke, etc.",
    "scam, phony, etc.",
    "mistake, error, etc.",
    "fabricated, misleading, etc.",
    "conspiracy, dubious, etc.",
    "satire, humor, etc.",
    "fiction, purely, etc.",
    "clickbait, sensational, etc.",
]
cluster_labels = [
    r"$\bf{legend}$, $\bf{tale}$, etc.",
    "altered, digitally, etc.",
    "hoax, joke, etc.",
    "scam, phony, etc.",
    "mistake, error, etc.",
    "fabricated, misleading, etc.",
    "conspiracy, dubious, etc.",
    "satire, humor, etc.",
    "fiction, purely, etc.",
    "clickbait, sensational, etc.",
]
color_dict = {
    "legend, tale, etc.": "#1f77b4",
    "altered, digitally, etc.": "#ff7f0e",
    "hoax, joke, etc.": "#2ca02c",
    "scam, phony, etc.": "#d62728",
    "mistake, error, etc.": "#9467bd",
    "fabricated, misleading, etc.": "#8c564b",
    "conspiracy, dubious, etc.": "#e377c2",
    "satire, humor, etc.": "#7f7f7f",
    "fiction, purely, etc.": "#bcbd22",
    "clickbait, sensational, etc.": "#17becf",
}
event_pairs = {
    "election": [
        "’16 US election",
        "’20 US election",
    ],
    "virus": [
        "H1N1",
        "COVID-19",
    ]
}
markers = ["^v7", "Ds6"]


def bebold(s):
    w1 = s.split(", ")[0]
    w2 = s.split(", ")[1]
    return r"$\bf{}$, $\bf{}$, etc.".format(w1, w2)


# Plot path settings.
figs_path = "figs"
if not os.path.exists(figs_path):
    os.mkdir(figs_path)


# Plot event wise diff.
def plot_event(event_pairs, df):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 6))
    for i, (event_pair, events) in enumerate(event_pairs.items()): 
        e0, e1 = events[0], events[1]
        tdf0, tdf1 = df[df[events[0]]], df[df[events[1]]]
        if "election" in event_pair:
            tdf0 = tdf0[tdf0["year"] == 2016]
            tdf1 = tdf1[tdf1["year"] == 2020]
        else:
            tdf0 = tdf0[tdf0["year"] < 2013]
            tdf1 = tdf1[tdf1["year"] > 2018]
        total0 = len(tdf0)
        total1 = len(tdf1)
        for j, cluster in enumerate(clusters):
            pos0 = len(tdf0[tdf0[cluster]])
            low0, high0 = pc(pos0, total0)
            pos1 = len(tdf1[tdf1[cluster]])
            low1, high1 = pc(pos1, total1)
        
            if low0 > high1 or low1 > high0:
                ax[i].axhline(j, lw=35, color="#D8D8D8")
            d = 0.15
            ax[i].plot([low0, high0], [j-d, j-d],
                       color=color_dict[cluster], lw=2, alpha=0.5)
            ax[i].plot([pos0 / total0], [j-d], color=color_dict[cluster],
                       marker=markers[i][0], markersize=int(markers[i][2]))
            ax[i].plot([low1, high1], [j+d, j+d],
                       color=color_dict[cluster], lw=2, alpha=0.5)
            ax[i].plot([pos1 / total1], [j+d], color=color_dict[cluster],
                       marker=markers[i][1], markersize=int(markers[i][2]))
            ax[i].set_ylim([9.5, -0.5])
            ax[i].set_xlim([0, 0.4])
            ax[i].set_xticks([0, 0.2, 0.4])
            ax[i].set_xticklabels(["     0%", "20%", "40%      "])
        ax[i].plot(-1, -1, color="k", label=e0,
                   marker=markers[i][0], markersize=int(markers[i][2]))
        ax[i].plot(-1, -1, color="k", label=e1,
                   marker=markers[i][1], markersize=int(markers[i][2]))
        leg = ax[i].legend(bbox_to_anchor=(0.5, 1.05, 0, 0), loc="lower center", borderaxespad=0.)
        leg.get_frame().set_alpha(0)
    ax[0].set_yticks([9-i for i in range(10)])
    ax[0].set_yticklabels([bebold(clusters[9-i]) for i in range(10)])
    ax[1].set_yticks([])
    ax[1].set_yticklabels([])
    for edge in ["right", "left", "top", "bottom"]:
        ax[0].spines[edge].set_visible(False)
        ax[1].spines[edge].set_visible(False)
    plt.savefig(os.path.join(figs_path, "events.pdf"),
                bbox_inches="tight", pad_inches=0)


# Plot cluster per year.
def plot_cluster(clusters, df):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(24, 5))
    for i, cluster in enumerate(clusters):
        ax = axes[i // 5][i % 5]
        years, rates = [], []
        for t in [0, 0.2, 0.4]:
            ax.axhline(t, linestyle="--", lw=1, color="#999999")
        for year, tdf in df.groupby("year"):
            years.append(year)
            pos = len(tdf[tdf[cluster]])
            total = len(tdf)
            low, high = pc(pos, total)
            ax.plot([year, year], [low, high],
                    color=color_dict[cluster], lw=2, alpha=0.5)
            rates.append(pos / total)
        ax.plot(years, rates, color=color_dict[cluster], lw=2, marker="o")
        ax.set_xlim([2009.5, 2020.5])
        if i // 5 == 0:
            ax.set_xticks([2010, 2012, 2014, 2016, 2018, 2020])
            ax.set_xticklabels(["’10", "’12", "’14", "’16", "’18", "’20"])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])
        ax.xaxis.tick_top()
        ax.set_xlabel(bebold(cluster))
        ax.set_ylim([-0.02, 0.58])
        if i % 5 == 0:
            ax.set_yticks([0, 0.2, 0.4])
            ax.set_yticklabels(["0%", "20%", "40%"])
        else:
            ax.set_yticks([])
            ax.set_yticklabels([])
        for edge in ["right", "left", "top", "bottom"]:
            ax.spines[edge].set_visible(False)
        plt.savefig(os.path.join(figs_path, "cluster.pdf"),
                    bbox_inches="tight", pad_inches=0)


def get_year(date):
    year = int(date[:4])
    if year < 1 or year > 2020:
        return np.nan
    return year if year > 2010 else 2010


# Read data.
df = pd.read_csv("data_w_rationale_clusters.csv")
df["year"] = df["date"].apply(get_year)
plot_event(event_pairs, df)
plot_cluster(clusters, df)
