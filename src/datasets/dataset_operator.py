# coding: utf-8


import random


class SentenceClassificationSet(object):
    """
    Generic dataset operator for sentence classification tasks.
    """

    def __init__(self):
        self.instances = []
        self.label2instance_dict = {}
        
    def add_one(self, tokens, label):
        self.instances.append({"sentence": " ".join(tokens), "label": label})
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


class SentenceClassificationSetSplit(SentenceClassificationSet):
    """
    Split to train/dev set of SentenceClassificationSet.
    Outputs:
        data_set_larger -- train set.
        data_set_smaller -- dev set.
    """

    def __init__(self):
        super(SentenceClassificationSetSplit, self).__init__()

    def split_datasets(self, ratio):
        data_set_larger = SentenceClassificationSet()
        data_set_smaller = SentenceClassificationSet()
        for instance in self.instances:
            if random.random() > ratio:
                data_set_pointer = data_set_larger
            else:
                data_set_pointer = data_set_smaller

            label = instance["label"]

            data_set_pointer.instances.append(instance)
            if label not in data_set_pointer.label2instance_dict:
                data_set_pointer.label2instance_dict[label] = {}

            data_set_pointer.label2instance_dict[label][len(data_set_pointer.instances)] = 1

        return data_set_larger, data_set_smaller
