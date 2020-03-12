# coding: utf-8


import os, json, gzip, random


class DataCleaner(object):
    """
    Dataset cleaner for beer reviews.
    """

    def __init__(self, data_dir="raw", score_threshold=0.6, split_ratio=0.1):
        """
        Inputs:
            data_dir -- the directory of the dataset.
            score_threshold -- the threshold (0-1) for pos/neg.
            split_ratio -- split ratio for train/dev.
        """
        self.data_dir = data_dir
        self.score_threshold = score_threshold
        self.split_ratio = split_ratio
        self.label_vocab = {}
        

    def clean(self):
        with open(os.path.join(self.data_dir, "sec_name_dict.json"), "r") as filtered:
            self.filtered_name_dict = json.load(filtered)  # Words to be filtered.
        
        train_path = os.path.join(self.data_dir, "reviews.aspect0.train.txt.gz")
        outfile = (open("train.tsv", "w"), open("dev.tsv", "w"))
        self._helper(train_path, outfile, split_ratio=self.split_ratio)
        print("Train/Dev is cleaned with %.2f" % self.split_ratio)

        test_path = os.path.join(self.data_dir, "reviews.aspect0.heldout.txt.gz")
        outfile = (open("test.tsv", "w"), None)
        self._helper(test_path, outfile)
        print("Test data is cleaned.") 
        

    def _helper(self, infile, outfile, split_ratio=None):
        with gzip.open(os.path.join(infile), "r") as f:
            for idx, line in enumerate(f):

                # Parser for beer reviews and get a single aspect.
                lbl, txt = tuple(line.decode("utf-8").strip("\n").split("\t"))
                lbl = float(lbl.split(" ")[0])  # 0 for appearance.
                
                if lbl > self.score_threshold:
                    label = "positive"
                else:
                    label = "negative"
                    
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)
                    label = self.label_vocab[label]
                else:
                    label = self.label_vocab[label]
                
                txt = txt.split()
                tokens = [term.lower() for term in txt if term != ""]
                
                start = -1
                for i, token in enumerate(tokens):
                    if token == ":" and i > 0:
                        name = tokens[i-1]
                        if name == "a" or name == "appearance":
                            start = i - 1
                            break
                if start < 0:
                    continue
                
                end = -1
                for i, token in enumerate(tokens):
                    if i <= start + 1:
                        continue
                    if token == ":" and i > 0:
                        name = tokens[i-1]
                        if name in self.filtered_name_dict:
                            end = i - 1
                            break
                if end < 0:
                    continue

                # Write one line.
                rationale = "0" * (end - start)
                tokens = " ".join(tokens[start:end])
                label = str(label)
                line = "\t".join([label, tokens, rationale]) + "\n"
                if split_ratio:
                    if random.random() > split_ratio:
                        outfile[0].write(line)
                    else:
                        outfile[1].write(line)
                else:
                    outfile[0].write(line)


if __name__ == "__main__":
    random.seed(0)
    DataCleaner().clean()
