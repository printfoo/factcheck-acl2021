# coding: utf-8


import os, shutil, json
import numpy as np


# Find the best checkpoint.
def find_best_ckpt(path, metric="f1", by="dev"):
    if not os.path.exists(path):
        return "[Checkpoints not found.]"
    with open(os.path.join(path, "record.json"), "r") as f:
        record = json.load(f)
    ckpts = [f for f in os.listdir(path) if f.endswith(".pt")]
    ckpts.sort()
    if len(record[by]) != len(ckpts):
        return "[Checkpoints unexpected error.]"
    best_ckpt_index = np.nanargmax([_["prediction"][metric] for _ in record[by]])
    best_ckpt_path = os.path.join(path, ckpts[best_ckpt_index])
    return best_ckpt_path


# Initialize checkpoint path.
def init_ckpt(path):
    if not os.path.exists(path):
        os.mkdir(path)
        with open(os.path.join(path, "README.md"), "w") as f:
            f.write("# Checkpoints.")


# Purge all saved checkpoints.
def purge(path):
    if os.path.exists(path):
        shutil.rmtree(path)
