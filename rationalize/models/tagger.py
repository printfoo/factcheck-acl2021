# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.nn import RnnModel


class Tagger(nn.Module):
    """
    Tagger module, input sequence and output binary mask.
    """

    def __init__(self, args, input_dim):
        """
        Inputs:
            args.z_dim -- rationale, always 2.
            args.hidden_dim -- dimension of hidden states.
            args.embedding_dim -- dimension of word embeddings.
            args.layer_num -- number of RNN layers.
            args.cell_type -- type of RNN cells, "GRU" or "LSTM".
        """
        super(Tagger, self).__init__()
        self.tagger = RnnModel(args, input_dim)
        self.output = nn.Linear(args.hidden_dim, args.z_dim)

    def forward(self, x, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim).
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length).
        """ 
        hiddens = self.tagger(x, mask).transpose(1, 2).contiguous()  # (batch_size, sequence_length, hidden_dim)
        z = self.output(hiddens)  # (batch_size, sequence_length, 2)
        return z
