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

    def __init__(self, data_dir, args):
        """
        Initialize a dataset for sentence classification:
        Inputs:
            data_dir -- the directory of the dataset.
            args.truncate_num -- max length for tokens.
            args.freq_threshold -- min frequency for tokens.
        """
        self.data_dir = data_dir
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
        self.idx_2_word = {val: key for key, val in self.word_vocab.items()}
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
        file_path = os.path.join(self.data_dir, data_set + ".tsv")
        with open(file_path, "r") as f:
            for line in f:
                label, tokens, rationale = line.split("\t")
                label = int(label)
                if label not in self.label_vocab:
                    self.label_vocab[label] = label
                tokens = tokens.split(" ")
                self.data_sets[data_set].add_one(tokens, label, truncate_num=self.truncate_num)


    def initial_embedding(self, embedding_size, embedding_path=None):
        """
        This function initialize embedding with glove embedding.
        If a word has embedding in glove, use the glove one.
        If not, initial with random.
        Inputs:
            embedding_size -- the dimension of the word embedding.
            embedding_path -- the path to the glove embedding file.
        Outputs:
            embeddings -- a numpy matrix in shape of (vocab_size, embedding_dim),
                          the ith row indicates the word with index i from word_ind_dict.
        """

        # Initialize a numpy embedding matrix.
        embeddings = 0.1 * np.random.randn(len(self.word_vocab), embedding_size).astype(np.float32)

        # Replace <PAD> by all zeros.
        embeddings[self.word_vocab["<PAD>"], :] = np.zeros(embedding_size, dtype=np.float32)

        # Load pre-trained embeddings if specified.
        if not embedding_path:
            print("No specified embedding file.")
            print("Embedding are randomly initialized.")
        elif not os.path.isfile(embedding_path):
            print("Specified embedding file does not exist:", embedding_path)
            print("Embedding are randomly initialized.")
        else:
            print("Loading embeddings from:", embedding_path)
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
            print("%d embeddings are initialized." % counter)

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
            batch_size: an integer for barch size.
        Outputs:
            x_mat -- numpy array in shape of (batch_size, max length of the sequence in the batch).
            y_vec -- numpy array of binary labels, numpy array in shape of (batch_size,).
            x_mask -- numpy array in shape of (batch_size, max length of the sequence in the batch).
        """

        data_set = self.data_sets[set_id]
        xs_, ys_, max_x_len_ = data_set.get_samples_from_one_list(batch_idx, self.truncate_num)

        x_masks_ = []
        for i, x in enumerate(xs_):
            xs_[i] = x + (max_x_len_ - len(x)) * [0]
            x_masks_.append([1] * len(x) + [0] * (max_x_len_ - len(x)))
            
        x_mat = np.array(xs_, dtype=np.int64)
        x_mask = np.array(x_masks_, dtype=np.int64)
        y_vec = np.array(ys_, dtype=np.int64)
        
        if sort:  # Sort all according to q_length.
            x_length = np.sum(x_mask, axis=1)
            x_sort_idx = np.argsort(-x_length)
            x_mat = x_mat[x_sort_idx, :]
            x_mask = x_mask[x_sort_idx, :]
            y_vec = y_vec[x_sort_idx]
        
        return x_mat, y_vec, x_mask


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
            word = self.idx_2_word[word_index]
            if display_flag:
                output_word = "%s %s%s" % (fg(1), word, attr(0))
                print(output_word, end="")
            else:
                print(" " + word, end="")
        print()


# Test DataLoader.
def test_data(data_dir, args):
    dataloader = SentenceClassification(data_dir, args)
    print(dataloader)
