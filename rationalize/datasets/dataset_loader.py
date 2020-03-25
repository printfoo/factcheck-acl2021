# coding: utf-8


import random, sys, os
import numpy as np
from colored import fg, attr, bg

from datasets.dataset_operator import SentenceClassificationSet


class SentenceClassification(object):
    """
    Generic dataset loader for sentence classification tasks.
    Functions need overwriting for a specific dataset.
    """

    def __init__(self, data_path, args):
        """
        Initialize a dataset for sentence classification:
        Inputs:
            data_path -- the directory of the dataset.
            args.truncate_num -- max length for tokens.
            args.freq_threshold -- min frequency for tokens.
        """
        self.data_path = data_path
        self.truncate_num = args.truncate_num
        self.freq_threshold = args.freq_threshold
        
        self.word_vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.label_vocab = {}

        print("Loading dataset.")
        self.data_sets = {"train": None, "dev": None, "test": None}
        for data_set in self.data_sets:
            self.load_dataset(data_set)
            self.data_sets[data_set].print_info()

        print("Building vocabulary.")
        self._build_vocab()

        print("Converting token to indexes.")
        self.idx2word = {val: key for key, val in self.word_vocab.items()}
        self.idx2label = {val: key for key, val in self.label_vocab.items()}


    def _build_vocab(self):
        """
        Filter the vocabulary and index words.
        This stores:
            data_set.pairs -- a list of [{"sentence": [wid1, wid2, ...], "label": 1}, ...].
        """
        
        # Add vocab one by one from sentence.
        def _add_vocab_from_sentence(word_freq_dict, sentence):
            tokens = sentence.split(" ")
            word_idx_list = []
            for token in tokens:
                if word_freq_dict[token] < self.freq_threshold:
                    word_idx_list.append(self.word_vocab["<UNK>"])
                else:
                    if token not in self.word_vocab:
                        self.word_vocab[token] = len(self.word_vocab)
                    word_idx_list.append(self.word_vocab[token])
            return word_idx_list
        
        # Index words in sentence for training pairs.
        def _index_words(word_freq_dict, pairs):
            ret_pair_list = []
            for pair_dict_ in pairs:
                new_pair_dict_ = {}
                
                for k, v in pair_dict_.items():
                    if k == "sentence":
                        new_pair_dict_[k] = _add_vocab_from_sentence(word_freq_dict, v)
                    else:
                        new_pair_dict_[k] = pair_dict_[k] 
                
                ret_pair_list.append(new_pair_dict_)
            return ret_pair_list
        
        word_freq_dict = self._get_word_freq(self.data_sets)
            
        for data_id, data_set in self.data_sets.items():
            data_set.pairs = _index_words(word_freq_dict, data_set.get_pairs())

        print('Size of the final vocabulary:', len(self.word_vocab))
        
        
    def _get_word_freq(self, data_sets_):
        """
        Build word frequency dictionary from sentence pairs.
        Outputs:
            word_freq_dict -- raw vocabulary.
        """

        # Add vocab one by one from sentence.
        def _add_freq_from_sentence(word_freq_dict, sentence):
            tokens = sentence.split(" ")
            for token in tokens:
                if token not in word_freq_dict:
                    word_freq_dict[token] = 1
                else:
                    word_freq_dict[token] += 1

        word_freq_dict = {}

        for data_id, data_set in data_sets_.items():
            for pair_dict in data_set.get_pairs():
                sentence = pair_dict["sentence"]
                _add_freq_from_sentence(word_freq_dict, sentence)

        print('Size of the raw vocabulary:', len(word_freq_dict))
        return word_freq_dict


    def load_dataset(self, data_set):
        """
        Load dataset and store to self.data_sets.
        Inputs:
            data_set -- the name of the dataset, train/dev/test.
        """

        # Load instances.
        self.data_sets[data_set] = SentenceClassificationSet()
        file_path = os.path.join(self.data_path, data_set + ".tsv")
        with open(file_path, "r") as f:
            for line in f:
                label, tokens = line.split("\t")[:2]
                if label not in self.label_vocab:
                    self.label_vocab[label] = len(self.label_vocab)
                label = self.label_vocab[label]
                tokens = tokens.split(" ")
                self.data_sets[data_set].add_one(tokens, label, truncate_num=self.truncate_num)


    def initial_embedding(self, method="random", size=100, path=None):
        """
        This function initialize embedding with glove embedding.
        If a word has embedding in glove, use the glove one.
        If not, initial with random.
        Inputs:
            method -- the method for embedding, onehot/random/pretrained.
            size -- the dimension of the word embedding, ignored if method==random.
            path -- the path to the embedding file, if method==pretrained.
        Outputs:
            embeddings -- a numpy matrix in shape of (vocab_size, embedding_dim),
                          the ith row indicates the word with index i from word_ind_dict.
        """
        
        if method in {"random", "pretrained"}:  # Fixed-size embedding.
            embeddings = 0.1 * np.random.randn(len(self.word_vocab), size).astype(np.float32)  # Random.
            embeddings[self.word_vocab["<PAD>"], :] = np.zeros(size, dtype=np.float32)  # <PAD>=0
            if method == "random":
                return embeddings
            else:  # Load pre-trained embeddings if specified.
                with open(path, "r") as f:
                    print("Loading embeddings from:", path)
                    for line in f:
                        data = line.strip().split(" ")
                        word = data[0].strip()
                        if word in self.word_vocab:
                            embeddings[self.word_vocab[word], :] = list(map(np.float32, data[1::]))
                return embeddings
        elif method in {"onehot"}:  # One-hot embedding of word_vocab size.
            embeddings = np.zeros((len(self.word_vocab), len(self.word_vocab))).astype(np.float32)  # One-hot.
            for word in self.word_vocab:
                if word != "<PAD>": # <PAD>=0
                    embeddings[self.word_vocab[word], self.word_vocab[word]] = 1
            return embeddings


    def get_train_batch(self, batch_size, sort=False):
        """
        Randomly sample a batch to train.
        Inputs:
            batch_size: an integer for barch size.
        Outputs:
            x_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch).
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,).
            x_mask -- numpy array in shape of (batch_size, max length of the sequence in the batch).
        """
        set_id = "train"
        data_set = self.data_sets[set_id]
        batch_idx = np.random.randint(0, data_set.size(), size=batch_size)
        return self.get_batch(set_id, batch_idx, sort)


    def get_batch(self, set_id, batch_idx, sort=False):
        """
        Randomly sample a batch to train.
        Inputs:
            batch_size: an integer for batch size.
        Outputs:
            x -- numpy array of input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            y -- numpy array of label y, shape (batch_size,),
                 only one element per instance 0-|label| pointing to a label.
            m -- numpy array of mask m, shape (batch_size, seq_len).
                 each element in the seq_len is of 0/1 selecting a token or not.
        """

        data_set = self.data_sets[set_id]
        xs_, ys_, max_x_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        ms_ = []
        for i, x in enumerate(xs_):
            xs_[i] = x + (max_x_len_ - len(x)) * [0]
            ms_.append([1] * len(x) + [0] * (max_x_len_ - len(x)))  # Mask <PAD>.
            
        x = np.array(xs_, dtype=np.int64)
        y = np.array(ys_, dtype=np.int64)
        m = np.array(ms_, dtype=np.int64)
        
        if sort:  # Sort all according to seq_len.
            x_sort_idx = np.argsort(-np.sum(m, axis=1))
            x = x[x_sort_idx, :]
            y = y[x_sort_idx]
            m = m[x_sort_idx, :]

        return x, y, m


    def display_example(self, x, z=None, threshold=0.9):
        """
        Display sentences and rationales.
        Inputs:
            x -- input sequence of word indices, (sequence_length,).
            z -- input rationale sequence, (sequence_length,).
            threshold -- display as rationale if z_i >= threshold.
        """
        condition = z >= threshold
        for word_index, display_flag in zip(x, condition):
            word = self.idx2word[word_index]
            if display_flag:
                output_word = "%s %s%s" % (fg(1), word, attr(0))
                print(output_word, end="")
            else:
                print(" " + word, end="")
        print()


# Test DataLoader.
def test_data(data_path, args):
    # print(dataloader.word_vocab)
    # print(dataloader.label_vocab)
    data = SentenceClassification(data_path, args)  # Load data.
    args.num_labels = len(data.label_vocab)  # Number of labels.
    embeddings = data.initial_embedding("random", 100)  # Load embeddings.
    embeddings = data.initial_embedding("onehot", 100)  # Load embeddings.
    print(embeddings)
