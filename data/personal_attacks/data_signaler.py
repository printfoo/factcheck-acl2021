# coding: utf-8


import os, json
import pandas as pd


class DataSignaler(object):
    """
    Dataset signaler for personal attacks.
    Add signals to train.tsv, dev.tsv and test.tsv.
    """

    def __init__(self):
        self.data_dirs = ["train.tsv", "dev.tsv", "test.tsv"]
        self.vocab_dir = os.path.join("linear_bow.analyze", "word_weight.json")
        vocab = pd.read_json(self.vocab_dir, lines=True)
        self.signal_dicts = {}
        for l in set(vocab.columns) - {"word"}:
            self.signal_dicts[l] = {w: s for w, s in vocab[["word", l]].values}
            print(l, "fuck", self.signal_dicts[l]["fuck"])
        signal_sorted = sorted(self.signal_dicts[l].items(), key=lambda _: abs(_[1]))
        threshold = signal_sorted[int(len(signal_sorted) * -0.01):]
        print("Threshold:", threshold[0][1])
    

    def _get_signal(self, row):
        signal_dict = self.signal_dicts[row["label"]]
        comment = row["tokens"].split(" ")
        signal = ["{:.5f}".format(signal_dict[c])
                  if c in signal_dict else "0.0" for c in comment]
        return " ".join(signal)
            
    
    def signal(self, data_dir):
        df = pd.read_csv(data_dir, sep="\t")
        df["linear_signal"] = df.apply(self._get_signal, axis=1)
        df.to_csv(data_dir, index=False, sep="\t")


if __name__ == "__main__":
    signaler = DataSignaler()
    signaler.signal(signaler.data_dirs[0])
    signaler.signal(signaler.data_dirs[1])
    signaler.signal(signaler.data_dirs[2])
