# coding: utf-8


import os, gzip, json

from datasets.dataset_loader import SentenceClassification
from datasets.dataset_operator import SentenceClassificationSet, SentenceClassificationSetSubSampling


class BeerReviewsSingleAspectWithTest(SentenceClassification):
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1, aspect=0, score_threshold=0.5, split_ratio=0.15):
        """
        This function initialize a dataset from Beer Review:
        Inputs:
            data_dir -- the directory containing the data
            aspect -- an integer of an aspect from 0-4
            truncate_num -- max length of the review text to use
        """
        self.aspect = aspect
        self.score_threshold = score_threshold
        self.aspect_names = ['apperance', 'aroma', 'palate', 'taste']
        self.split_ratio = split_ratio

        super(BeerReviewsSingleAspectWithTest, self).__init__(data_dir, truncate_num, freq_threshold)
        
        
    def load_dataset(self):
        
        filein = open(os.path.join(self.data_dir, 'sec_name_dict.json'), 'r')
        self.filtered_name_dict = json.load(filein)
        filein.close()
        
        self.data_sets = {}
        
        # load train
        tmp_dataset = self._load_data_set(os.path.join(self.data_dir, 
                                                       'reviews.aspect{:d}.train.txt.gz'.format(self.aspect)),
                                          with_dev=True)
        
        print('splitting with %.2f'%self.split_ratio)
        self.data_sets['train'], self.data_sets['dev'] = tmp_dataset.split_datasets(self.split_ratio)
        self.data_sets['train'].print_info()
        self.data_sets['dev'].print_info()
        
        # load dev
        self.data_sets['test'] = self._load_data_set(os.path.join(self.data_dir, 
                                                                 'reviews.aspect{:d}.heldout.txt.gz'.format(self.aspect)))
    
        # build vocab
        self._build_vocab()
        
        self.idx2label = {val: key for key, val in self.label_vocab.items()}
        
        
    def _load_data_set(self, fpath, with_dev=False):
        """
        Inputs: 
            fpath -- the path of the file. 
        Outputs:
            positive_pairs -- a list of positive question-passage pairs
            negative_pairs -- a list of negative question-passage pairs
        """
        
        if with_dev:
            data_set = SentenceClassificationSetSubSampling()
        else:
            data_set = SentenceClassificationSet()
        
        section_name_dict = {}
        
        with gzip.open(os.path.join(fpath), 'r') as f:
            for idx, line in enumerate(f):
                lbl, txt = tuple(line.decode('utf-8').strip('\n').split('\t'))
                lbl = float(lbl.split(' ')[self.aspect])
                
                if lbl > self.score_threshold:
                    label = 'positive'
                else:
                    label = 'negative'
                    
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)
                    label = self.label_vocab[label]
                else:
                    label = self.label_vocab[label]
                
                txt = txt.split()[:self.truncate_num]
                tokens = [term.lower() for term in txt if term != '']
                
                start = -1
                for i, token in enumerate(tokens):
                    if token == ':' and i > 0:
                        name = tokens[i-1]
                        
                        if name == 'a' or name == 'appearance':
                            start = i - 1
                            break
                
                if start < 0:
                    continue
                
                end = -1
                for i, token in enumerate(tokens):
                    if i <= start + 1:
                        continue
                    if token == ':' and i > 0:
                        name = tokens[i-1]
                        
                        if name in self.filtered_name_dict:
                            end = i - 1
                            break
                            
                if end < 0:
                    continue

                data_set.add_one(tokens[start:end], label)
            
        data_set.print_info()

        return data_set
