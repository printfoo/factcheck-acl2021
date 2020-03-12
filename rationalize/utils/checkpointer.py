# coding: utf-8


import os, shutil


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
