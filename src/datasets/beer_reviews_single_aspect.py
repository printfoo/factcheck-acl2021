# coding: utf-8


import os, gzip, json

from datasets.dataset_loader import SentenceClassification
from datasets.dataset_operator import SentenceClassificationSet, SentenceClassificationSetSubSampling


class BeerReviewsSingleAspectWithTest(SentenceClassification):
    """
    Dataset loader for beer reviews.
    Inherited from datasets.dataset_loader.SentenceClassification.
    """

    def __init__(self, data_dir, truncate_num=300, freq_threshold=1, aspect=0, score_threshold=0.5, split_ratio=0.15):
        """
        Initialize a dataset for beer reviews:
        Inputs:
            data_dir -- the directory of the dataset.
            truncate_num -- max length for tokens.
            freq_threshold -- min frequency for tokens.
            aspect -- an integer (0-4) for an aspect.
            score_threshold -- the threshold (0-1) for pos/neg.
            split_ratio -- split ratio for train/dev.
        """
        self.aspect = aspect
        self.score_threshold = score_threshold
        self.aspect_names = ["apperance", "aroma", "palate", "taste"]
        self.split_ratio = split_ratio
        self.data_sets = {}
        super(BeerReviewsSingleAspectWithTest, self).__init__(data_dir, truncate_num, freq_threshold)

        
    def load_dataset(self):
        """
        Load dataset and store to self.data_sets.
        """

        # Words that are not reviews and need to be filtered.
        with open(os.path.join(self.data_dir, "sec_name_dict.json"), "r") as filtered:
            self.filtered_name_dict = json.load(filtered)
        
        # Load train and dev sets.
        train_path = os.path.join(self.data_dir, "reviews.aspect{:d}.train.txt.gz".format(self.aspect))
        tmp_dataset = self._load_dataset_helper(train_path, with_dev=True)
        print('Splitting train/dev with %.2f' % self.split_ratio)
        self.data_sets["train"], self.data_sets["dev"] = tmp_dataset.split_datasets(self.split_ratio)
        
        # Load test set.
        test_path = os.path.join(self.data_dir, "reviews.aspect{:d}.heldout.txt.gz".format(self.aspect))
        self.data_sets["test"] = self._load_dataset_helper(test_path)
   
        # Print dataset info.
        for data_set in self.data_sets:
            print(data_set)
            self.data_sets[data_set].print_info()

        # Build vocab.
        self._build_vocab()
        
        
    def _load_dataset_helper(self, fpath, with_dev=False):
        """
        Helper to load data.
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            data_set -- a datasets.dataset_loader.SentenceClassificationSet object.
            data_set.instances -- a list of [{"sentence": "good", "label": 1}, ...]
        """
        
        if with_dev:  # Split to train and dev sets.
            data_set = SentenceClassificationSetSubSampling()
        else:  # Train set only.
            data_set = SentenceClassificationSet()
        
        with gzip.open(os.path.join(fpath), "r") as f:
            for idx, line in enumerate(f):

                # Parser for beer reviews and get a single aspect.
                lbl, txt = tuple(line.decode("utf-8").strip("\n").split("\t"))
                lbl = float(lbl.split(" ")[self.aspect])
                
                if lbl > self.score_threshold:
                    label = "positive"
                else:
                    label = "negative"
                    
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)
                    label = self.label_vocab[label]
                else:
                    label = self.label_vocab[label]
                
                txt = txt.split()[:self.truncate_num]
                tokens = [term.lower() for term in txt if term != '']
                
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

                # Add to the data_set object.
                data_set.add_one(tokens[start:end], label)

        return data_set
