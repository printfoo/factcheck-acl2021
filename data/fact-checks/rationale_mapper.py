# coding: utf-8


import os, json
import pandas as pd
import numpy as np


def get_cluster_name(cluster):
    top2 = []
    for w, f in sorted(cluster.items(), key=lambda kv: -kv[1]):
        if not top2:
            top2.append(w)
        else:
            if len(set(w).intersection(top2[0])) < 5 and len(w.split(" ")) == 1:
                top2.append(w)
                break
    return ", ".join(top2) + ", etc."


def map_rationales(t, d):
    for rationale in d:
        if rationale in t:
            return True
    return False


# Read rationale clusters.
clusters_path = os.path.join("soft_rationalizer_w_domain.cluster",
                            "misinfo", "clusters.json")
with open(clusters_path, "r") as f:
    clusters = f.read().split("\n")[:-1]


# Read data.
dfs = []
for set_name in ["train", "dev", "test"]:
    df = pd.read_csv(set_name + ".tsv", sep="\t")
    dfs.append(df)
df = pd.concat(dfs)


# Label event.
events = {
    "’16 US election": {"hillary", "clinton", "donald", "trump"},
    "’20 US election": {"donald", "trump", "joe", "biden"},
    "H1N1": {"flu", "influenza", "h1n1"},
    "COVID-19": {"covid", "coronavirus"},
}
for event_name, event in events.items():
    df[event_name] = df["tokens"].apply(lambda t: map_rationales(t, event))
    print(event_name, "\t", len(df[df[event_name]]))


# Map rationale clusters.
for cluster in clusters:
    cluster = json.loads(cluster)
    cluster_name = get_cluster_name(cluster)
    df[cluster_name] = df["tokens"].apply(lambda t: map_rationales(t, cluster))
    print(cluster_name, "\t", len(df[df[cluster_name]]))


# Save mapped data.
df.to_csv("data_w_rationale_clusters.csv", index=False)
