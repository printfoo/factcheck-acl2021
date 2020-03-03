# coding: utf-8


import os
import sys
import gzip
import random
import numpy as np
from colored import fg, attr, bg
import json


class TextDataset(object):
    """
    Generic text dataset loader.
    This class needs to be overwrite for specific tasks.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        
    def load_dataset(self):
        pass
    
    def initial_embedding(self, embedding_size, embedding_path=None):
        """
        This function initialize embedding with glove embedding. 
        If a word has embedding in glove, use the glove one.
        If not, initial with random.
        Inputs:
            embedding_size -- the dimension of the word embedding
            embedding_path -- the path to the glove embedding file
        Outputs:
            embeddings -- a numpy matrix in shape of (vocab_size, embedding_dim)
                          the ith row indicates the word with index i from word_ind_dict
        """    
        vocab_size = len(self.word_vocab)
        # initialize a numpy embedding matrix 
        embeddings = 0.1*np.random.randn(vocab_size, embedding_size).astype(np.float32)
        # replace <PAD> by all zero
        embeddings[0, :] = np.zeros(embedding_size, dtype=np.float32)

        if embedding_path and os.path.isfile(embedding_path):
            f = open(embedding_path, "r")
            counter = 0
            for line in f:
                data = line.strip().split(" ")
                word = data[0].strip()
                embedding = data[1::]
                embedding = list(map(np.float32, embedding))
                if word in self.word_vocab:
                    embeddings[self.word_vocab[word], :] = embedding
                    counter += 1
            f.close()
            print("%d words has been switched."%counter)
        else:
            print("embedding is initialized fully randomly.")
            
        return embeddings
    
    def data_to_index(self):
        pass
    
    def _index_to_word(self):
        """
        Apply reverse operation of word to index.
        """
        return {value: key for key, value in self.word_vocab.items()}
    
    def get_batch(self, dataset_id, batch_idx):
        """
        randomly select a batch from a dataset
        
        """
        pass
    
    
    def get_random_batch(self, dataset_id, batch_size):
        """
        randomly select a batch from the training set
        Inputs:
            dataset_id: a integer index of the dataset to sample, 0: train, 1: validation, 2: test
            batch_size: integer
        """
        pass
    
    
    def display_example(self, x, z=None, threshold=0.9):
        """
        Given word a suquence of word index, and its corresponding rationale,
        display it
        Inputs:
            x -- input sequence of word indices, (sequence_length,)
            z -- input rationale sequence, (sequence_length,)
            threshold -- display as rationale if z_i >= threshold
        Outputs:
            None
        """
        # apply threshold
        condition = z >= threshold
        for word_index, display_flag in zip(x, condition):
            word = self.idx_2_word[word_index]
            if display_flag:
                output_word = "%s %s%s" %(fg(1), word, attr(0))
                sys.stdout.write(output_word)                
            else:
                sys.stdout.write(" " + word)
        sys.stdout.flush()


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


class SentenceClassification(TextDataset):
    
    def __init__(self, data_dir, truncate_num=300, freq_threshold=1):        
        super(SentenceClassification, self).__init__(data_dir)
        self.truncate_num = truncate_num
        self.freq_threshold = freq_threshold
        
        self.word_vocab = {'<PAD>':0, '<START>':1, '<END>':2, '<UNK>':3}
        self.label_vocab = {}
        self.load_dataset()
        print('Converting text to word indicies.')
        self.idx_2_word = self._index_to_word()
        
    
    def _build_vocab(self):
        """
        Filter the vocabulary and numeralization
        """
        
        def _add_vocab_from_sentence(word_freq_dict, sentence):
            tokens = sentence.split(' ')
            word_idx_list = []
            for token in tokens:
                if word_freq_dict[token] < self.freq_threshold:
                    word_idx_list.append(self.word_vocab['<UNK>'])
                else:
                    if token not in self.word_vocab:
                        self.word_vocab[token] = len(self.word_vocab)
                    word_idx_list.append(self.word_vocab[token])
            return word_idx_list
        
        # numeralize passages in training pair lists
        def _numeralize_pairs(word_freq_dict, pairs):
            ret_pair_list = []
            for pair_dict_ in pairs:
                new_pair_dict_ = {}
                
                for k, v in pair_dict_.items():
                    if k == 'sentence':
                        new_pair_dict_[k] = _add_vocab_from_sentence(word_freq_dict, v)
                    else:
                        new_pair_dict_[k] = pair_dict_[k] 
                
                ret_pair_list.append(new_pair_dict_)
            return ret_pair_list
        
        
        word_freq_dict = self._get_word_freq(self.data_sets)
            
        for data_id, data_set in self.data_sets.items():
            data_set.pairs = _numeralize_pairs(word_freq_dict, data_set.get_pairs())

        print('size of the final vocabulary:', len(self.word_vocab))
        
        
    def _get_word_freq(self, data_sets_):
        """
        Building word frequency dictionary and filter the vocabulary
        """
        
        def _add_freq_from_sentence(word_freq_dict, sentence):
            tokens = sentence.split(' ')
            for token in tokens:
                if token not in word_freq_dict:
                    word_freq_dict[token] = 1
                else:
                    word_freq_dict[token] += 1

        word_freq_dict = {}

        for data_id, data_set in data_sets_.items():
            for pair_dict in data_set.get_pairs():
                sentence = pair_dict['sentence']
                _add_freq_from_sentence(word_freq_dict, sentence)

        print('size of the raw vocabulary:', len(word_freq_dict))
        return word_freq_dict
    
    
    def get_train_batch(self, batch_size, sort=False):
        """
        randomly select a batch from a dataset
        Inputs:
            batch_size: 
        Outputs:
            q_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            p_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch)
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,)
        """
        
        set_id = 'train'
        data_set = self.data_sets[set_id]
        batch_idx = np.random.randint(0, data_set.size(), size=batch_size)
        
        return self.get_batch(set_id, batch_idx, sort)
    
    def get_batch(self, set_id, batch_idx, sort=False):
        
        data_set = self.data_sets[set_id]
        xs_, ys_, max_x_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        x_masks_ = []
        for i, x in enumerate(xs_):
            xs_[i] = x + (max_x_len_ - len(x)) * [0]
            x_masks_.append([1] * len(x) + [0] * (max_x_len_ - len(x)))
            
        x_mat = np.array(xs_, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        y_vec = np.array(ys_, dtype=np.int64)
        
        if sort:
            # sort all according to q_length
            x_length = np.sum(x_mask, axis=1)
            x_sort_idx = np.argsort(-x_length)
            x_mat = x_mat[x_sort_idx, :]
            x_mask = x_mask[x_sort_idx, :]
            y_vec = y_vec[x_sort_idx]
        
        return x_mat, y_vec, x_mask

    
    def display_sentence(self, x):
        """
        Display a suquence of word index
        Inputs:
            x -- input sequence of word indices, (sequence_length,)
        Outputs:
            None
        """
        # apply threshold
        for word_index in x:
            word = self.idx_2_word[word_index]
            sys.stdout.write(" " + word)
        sys.stdout.write("\n")
        sys.stdout.flush()

