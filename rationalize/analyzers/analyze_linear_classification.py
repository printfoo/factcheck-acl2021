# coding: utf-8


import torch
import os, json


def analyze(ckpt_path, out_path, data, n=float("inf")):

    model = torch.load(ckpt_path)  # Load model from checkpoint.    
    weight = model.linear.weight  # Weights of words.
    all_weight = torch.sum(weight, 0)  # Sum of weights across all labels.
    words_js = {}  # Output.

    for label_id in data.idx2label:
        label = data.idx2label[label_id]
        words_js[label] = {}
        print("Label:", label)

        # Weights for this label minus all others.
        label_weight = weight[label_id] * len(data.label_vocab) - all_weight

        # Sort descendingly.
        label_word_id = torch.argsort(label_weight, descending=True)

        # Display top n words.
        for i in range(min(n, len(label_word_id))):
            wid = label_word_id[i].item()
            word = data.idx2word[wid]
            word_w = label_weight[wid].item()
            words_js[label][word] = word_w

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    with open(os.path.join(out_path, "word_weight.json"), "w") as f:
        f.write(json.dumps(words_js))

