# coding: utf-8


import random


class ClassificationDataSet(object):
    """
    Generic dataset operator for sentence classification tasks.
    """

    def __init__(self):
        self.instances = []
        self.label2instance_dict = {}
        
    def add_one(self, tokens, label, rationale, signal, domain, truncate_num=0):
        if truncate_num > 0:  # Truncate sentences.
            tokens = tokens[:truncate_num]
            rationale = rationale[:truncate_num]
            signal = signal[:truncate_num]
            domain = domain[:truncate_num]
        self.instances.append({"tokens": tokens,
                               "label": label,
                               "rationale": rationale,
                               "signal": signal,
                               "domain": domain})
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
        ss = []
        ds = []
        max_x_len = -1

        for i, idx in enumerate(batch_idx):
            pair_dict_ = self.pairs[idx]

            # Add a label to sample.
            label = pair_dict_["label"]
            ys.append(label)
            
            # Add a list of tokens (of wids) to sample.
            tokens = pair_dict_["tokens"]
            if truncate_num > 0:  # Truncate sentences.
                tokens = tokens[:truncate_num]
            xs.append(tokens)
            
            # Add a list of rational annotation to sample.
            rationale = pair_dict_["rationale"]
            if truncate_num > 0:  # Truncate rationales.
                rationale = rationale[:truncate_num]
            rs.append(rationale)
            
            # Add a list of linear signal to sample.
            signal = pair_dict_["signal"]
            if truncate_num > 0:  # Truncate rationales.
                signal = signal[:truncate_num]
            ss.append(signal)
            
            # Add a list of domain knowledge to sample.
            domain = pair_dict_["domain"]
            if truncate_num > 0:  # Truncate rationales.
                domain = domain[:truncate_num]
            ds.append(domain)
             
            max_x_len = max(max_x_len, len(tokens))

        return xs, ys, rs, ss, ds, max_x_len
    
            
    def print_info(self):
        for k, v in self.label2instance_dict.items():
            print("Number of instances with label %d:" % k, len(v))
