# coding: utf-8


import torch


def analyze(ckpt_path, data, n=20):

    model = torch.load(ckpt_path)  # Load model from checkpoint.    
    weight = model.linear.weight  # Weights of words.
    all_weight = torch.sum(weight, 0)  # Sum of weights across all labels.

    for label_id in data.idx2label:
        print("Label:", data.idx2label[label_id])

        # Weights for this label minus all others.
        label_weight = weight[label_id] * len(data.label_vocab) - all_weight

        # Sort descendingly.
        label_word_id = torch.argsort(label_weight, descending=True)

        # Display top n words.
        for i in range(n):
            wid = label_word_id[i].item()
            print(wid, data.idx2word[wid], label_weight[wid].item())

