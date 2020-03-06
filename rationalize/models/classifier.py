# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.nn import CnnModel, RnnModel


class Classifier(nn.Module):
    """
    Classifier module, input sequence and binary mask, output label.
    Using CNN or RNN modules.
    """

    def __init__(self, args):
        """
        Inputs:
            args.num_labels -- number of labels.
            args.hidden_dim -- dimension of hidden states.
            args.embedding_dim -- dimension of word embeddings.
        """
        super(Classifier, self).__init__()
        self.NEG_INF = -1.0e6
        self.args = args
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.input_dim = args.embedding_dim

        self.encoder = RnnModel(self.args, self.input_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.num_labels)

    def forward(self, word_embeddings, z, mask):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """
        masked_input = word_embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, mask)

        max_hidden = torch.max(hiddens + (1 - mask * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
        predict = self.predictor(max_hidden)
        return predict
