# coding: utf-8


import os
from glob import glob


# Purge all saved checkpoints.
def purge(path):
    for f in glob(os.path.join(path, "*.ckpt")):
        os.remove(f)
    for f in glob(os.path.join(path, "*.json")):
        os.remove(f)
