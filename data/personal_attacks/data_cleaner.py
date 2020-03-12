# coding: utf-8


import os
import pandas as pd


class DataCleaner(object):
    """
    Dataset cleaner for beer reviews.
    """

    def __init__(self, data_dir="raw", score_threshold=0.6):
        """
        Inputs:
            data_dir -- the directory of the dataset.
            score_threshold -- the threshold (0-1) for pos/neg.
        """
        self.data_dir = data_dir
        self.score_threshold = score_threshold
        self.label_vocab = {}
        

    def clean(self):

        # Load sentence.
        sentence_path = os.path.join(self.data_dir, "attack_annotated_comments.tsv")
        sentence = pd.read_csv(sentence_path, delimiter="\t")

        # Load label.
        label_path = os.path.join(self.data_dir, "attack_annotations.tsv")
        label = pd.read_csv(label_path, delimiter="\t")
        label = label.groupby("rev_id").mean()

        # Merge sentence and label.
        df = sentence.merge(label, on="rev_id", how="inner")
        print(df)


if __name__ == "__main__":
    DataCleaner().clean()
