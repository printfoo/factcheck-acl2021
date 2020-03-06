# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.nn import CnnModel, RnnModel


class Generator(nn.Module):
    
    def __init__(self, args, input_dim):
        """
        args.z_dim -- rationale or not, always 2.
        args.model_type -- "CNN" or "RNN".

        if CNN:
            args.hidden_dim -- dimension of filters.
            args.embedding_dim -- dimension of word embeddings.
            args.kernel_size -- kernel size of the conv1d.
            args.layer_num -- number of CNN layers.      
        if use RNN:
            args.hidden_dim -- dimension of filters.
            args.embedding_dim -- dimension of word embeddings.
            args.layer_num -- number of RNN layers.
            args.cell_type -- type of RNN cells, "GRU" or "LSTM".
        """
        super(Generator, self).__init__()
        
        self.args = args
        self.z_dim = args.z_dim
        
        if args.model_type == "CNN":
            self.generator_model = CnnModel(args, input_dim)
        elif args.model_type == "RNN":
            self.generator_model = RnnModel(args, input_dim)
        self.output_layer = nn.Linear(args.hidden_dim, self.z_dim)

    def forward(self, x, mask=None):
        """
        Given input x in shape of (batch_size, sequence_length) generate a 
        "binary" mask as the rationale
        Inputs:
            x -- input sequence of word embeddings, (batch_size, sequence_length, embedding_dim).
        Outputs:
            z -- output rationale, "binary" mask, (batch_size, sequence_length).
        """
        
        # (batch_size, sequence_length, hidden_dim)
        hiddens = self.generator_model(x, mask).transpose(1, 2).contiguous() 
        scores = self.output_layer(hiddens) # (batch_size, sequence_length, 2)

        return scores
