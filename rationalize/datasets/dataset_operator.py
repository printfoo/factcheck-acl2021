# coding: utf-8


import random


class SentenceClassificationSet(object):
    """
    Generic dataset operator for sentence classification tasks.
    """

    def __init__(self):
        self.instances = []
        self.label2instance_dict = {}
        
    def add_one(self, sentence, label, rationale, truncate_num=0):
        if truncate_num > 0:  # Truncate sentences.
            sentence = sentence[:truncate_num]
            rationale = rationale[:truncate_num]
        self.instances.append({"sentence": " ".join(sentence),
                               "label": label,
                               "rationale": rationale})
        if label not in self.label2instance_dict:
            self.label2instance_dict[label] = {}
        
        self.label2instance_dict[label][len(self.instances)] = 1
        
    def get_pairs(self):
        return self.instances
    
    def size(self):
        return len(self.instances)
    
    def get_samples_from_ids(self, batch_idx, truncate_num=0):
        xs = []
        ys = []
        rs = []
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
            xs.append(sentence)
            
            # Add a rationale to sample.
            rationale = pair_dict_["rationale"]
            if truncate_num > 0:  # Truncate rationales.
                rationale = rationale[:truncate_num]
            rs.append(rationale)
             
            max_x_len = max(max_x_len, len(sentence))
            
        return xs, ys, rs, max_x_len
    
            
    def print_info(self):
        for k, v in self.label2instance_dict.items():
            print("Number of instances with label %d:" % k, len(v))
