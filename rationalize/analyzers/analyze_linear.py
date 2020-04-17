# coding: utf-8


import torch
import os, json


def analyze(ckpt_path, out_path, data):

    model = torch.load(ckpt_path)  # Load model from checkpoint.    
    weight = model.linear.weight  # Weights of words.

    if not os.path.exists(out_path):
        os.mkdir(out_path)
    f = open(os.path.join(out_path, "word_weight.json"), "w")

    for wid in range(len(weight[0])):  # Word id.
        word = data.idx2word[wid]  # Word.
        words_js = {"word": word}  # Output for a word.
        for label_id in data.idx2label:
            label = data.idx2label[label_id]  # Label.
            word_w = weight[label_id][wid].item()  # Word weight.
            words_js[label] = word_w
        f.write(json.dumps(words_js) + "\n")

    f.close()
