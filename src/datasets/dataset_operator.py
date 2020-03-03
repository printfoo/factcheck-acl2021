# coding: utf-8


import os
import sys
import gzip
import random
import numpy as np
from colored import fg, attr, bg
import json


class SentenceClassificationSet(object):
    '''
    '''
    def __init__(self):
        self.instances = []
        self.label2instance_dict = {}
        
    def add_one(self, tokens, label):
        self.instances.append({'sentence':' '.join(tokens), 'label':label})
        if label not in self.label2instance_dict:
            self.label2instance_dict[label] = {}
        
        self.label2instance_dict[label][len(self.instances)] = 1
        
    def get_pairs(self):
        return self.instances
    
    def size(self):
        return len(self.instances)
    
    def get_samples_from_one_list(self, batch_idx, truncate_num=0):
        xs = []
        ys = []
        max_x_len = -1

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            ys.append(label)
            
            sentence = pair_dict_['sentence']
            
            if truncate_num > 0:
                sentence = sentence[:truncate_num]
            if len(sentence) > max_x_len:
                max_x_len = len(sentence)
                
            xs.append(sentence)
            
        return xs, ys, max_x_len
    
    
    def get_ngram_samples_from_one_list(self, batch_idx, ngram=3, truncated_map=None):
        xs = []
        ys = []

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            sentence = pair_dict_['sentence']

            for start_idx in range(len(sentence) - (ngram - 1)):
                context_idxs = sentence[start_idx:start_idx+ngram-1]

                target_idx = sentence[start_idx+ngram-1]
#                 if truncated_map is not None:
#                     target_idx = truncated_map[target_idx]

                xs.append(context_idxs)
                ys.append(target_idx)

        return xs, ys
    
    def get_labeled_ngram_samples_from_one_list(self, trg_label, batch_idx, ngram=3, truncated_map=None):
        xs = []
        ys = []

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            
            if label != trg_label:
                continue
            
            sentence = pair_dict_['sentence']

            for start_idx in range(len(sentence) - (ngram - 1)):
                context_idxs = sentence[start_idx:start_idx+ngram-1]

                target_idx = sentence[start_idx+ngram-1]
#                 if truncated_map is not None:
#                     target_idx = truncated_map[target_idx]

                xs.append(context_idxs)
                ys.append(target_idx)

        return xs, ys
    
    
    def get_eval_samples_from_one_list(self, batch_idx, truncate_num=0):
        xs = []
        zs = []
        ys = []
        max_x_len = -1

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]
            label = pair_dict_['label']
            ys.append(label)
            
            sentence = pair_dict_['sentence']
            highlights = pair_dict_['z']
            
            if truncate_num > 0:
                sentence = sentence[:truncate_num]
                highlights = highlights[:truncate_num]
            if len(sentence) > max_x_len:
                max_x_len = len(sentence)
                
            xs.append(sentence)
            zs.append(highlights)
            
        return xs, ys, zs, max_x_len
    
            
    def print_info(self):
        for k, v in self.label2instance_dict.items():
            print('Number of instances with label%d:'%k, len(v))



class SentenceClassificationSetSubSampling(SentenceClassificationSet):
    '''
    '''
    def __init__(self):
        super(SentenceClassificationSetSubSampling, self).__init__()

    def split_datasets(self, ratio):
        data_set_larger = SentenceClassificationSet()
        data_set_smaller = SentenceClassificationSet()
        for instance in self.instances:
            if random.random() > ratio:
                data_set_pointer = data_set_larger
            else:
                data_set_pointer = data_set_smaller

            label = instance['label']

            data_set_pointer.instances.append(instance)
            if label not in data_set_pointer.label2instance_dict:
                data_set_pointer.label2instance_dict[label] = {}

            data_set_pointer.label2instance_dict[label][len(data_set_pointer.instances)] = 1

        return data_set_larger, data_set_smaller
