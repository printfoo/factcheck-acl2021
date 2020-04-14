# coding: utf-8


import torch
import torch.nn as nn
from torch.autograd import Variable  


class RnnModel(nn.Module):
    """
    Basic RNN module, input embedding and output hidden states.
    """

    def __init__(self, args, input_dim):
        """
        args.hidden_dim -- dimension of filters.
        args.embedding_dim -- dimension of word embeddings.
        args.layer_num -- number of RNN layers.
        args.cell_type -- type of RNN cells, GRU or LSTM.
        """
        super(RnnModel, self).__init__()
        
        self.args = args
 
        if args.cell_type == "GRU":
            self.rnn_layer = nn.GRU(input_size=input_dim, hidden_size=args.hidden_dim//2, 
                                    num_layers=args.layer_num, bidirectional=True)
        elif args.cell_type == "LSTM":
            self.rnn_layer = nn.LSTM(input_size=input_dim, hidden_size=args.hidden_dim//2, 
                                     num_layers=args.layer_num, bidirectional=True)
    
    def forward(self, embeddings, mask=None):
        """
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim).
            mask -- a float tensor of masks, (batch_size, length).
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length).
        """
        embeddings_ = embeddings.transpose(0, 1)  # (sequence_length, batch_size, embedding_dim)
        
        if mask is not None:
            seq_lengths = list(torch.sum(mask, dim=1).cpu().data.numpy())
            seq_lengths = list(map(int, seq_lengths))
            inputs_ = torch.nn.utils.rnn.pack_padded_sequence(embeddings_, seq_lengths)
        else:
            inputs_ = embeddings_
        
        hidden, _ = self.rnn_layer(inputs_)  # (sequence_length, batch_size, hidden_dim (* 2 if bidirectional))
        
        if mask is not None:
            hidden, _ = torch.nn.utils.rnn.pad_packed_sequence(hidden)  # (length, batch_size, hidden_dim)
        
        return hidden.permute(1, 2, 0)  # (batch_size, hidden_dim, sequence_length)
