# coding: utf-8


import random


class SentenceClassificationSet(object):
    """
    Generic dataset operator for sentence classification tasks.
    """

    def __init__(self):
        self.instances = []
        self.label2instance_dict = {}
        
    def add_one(self, sentence, label, truncate_num=0):
        if truncate_num > 0:  # Truncate sentences.
            sentence = sentence[:truncate_num]
        self.instances.append({"sentence": " ".join(sentence), "label": label})
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

            # Add a label to sample.
            label = pair_dict_["label"]
            ys.append(label)
            
            # Add a sentence (of wids) to sample.
            sentence = pair_dict_["sentence"]
            if truncate_num > 0:  # Truncate sentences.
                sentence = sentence[:truncate_num]
            max_x_len = max(max_x_len, len(sentence))
            xs.append(sentence)
            
        return xs, ys, max_x_len
    
            
    def print_info(self):
        for k, v in self.label2instance_dict.items():
            print("Number of instances with label %d:" % k, len(v))
