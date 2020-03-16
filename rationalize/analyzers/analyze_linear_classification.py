# coding: utf-8


import torch


def analyze(ckpt_path):

    model = torch.load(ckpt_path)  # Load model from checkpoint.    
    model.eval()  # Set model to eval mode.
    print(model)
