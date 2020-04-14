# coding: utf-8


import torch
import torch.nn as nn
from torch.autograd import Variable  


class RnnEncoder(nn.Module):
    """
    Basic RNN encoder module, input embeddings and output hidden states.
    """

    def __init__(self, args):
        """
        Inputs:
            args.hidden_dim -- dimension of hidden states.
            args.layer_num -- number of RNN layers.
            args.cell_type -- type of RNN cells, "GRU" or "LSTM".
            args.embedding_dim -- dimension of word embeddings.
        """
        super(RnnEncoder, self).__init__()
        cells = {"GRU": nn.GRU, "LSTM": nn.LSTM}
        self.rnn = cells[args.cell_type](input_size=args.embedding_dim,
                                         hidden_size=args.hidden_dim//2,
                                         num_layers=args.layer_num,
                                         bidirectional=True)
   

    def forward(self, e, m=None):
        """
        Inputs:
            e -- input sequence with embeddings, shape (batch_size, seq_len, embedding_dim),
                 each element in the seq_len is a word embedding of embedding_dim.
            m -- mask of the input sequence, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            hiddens -- hidden states of the encoder, shape (batch_size, hidden_dim, seq_len).
        """

        # Transpose embeddings,
        # (batch_size, seq_len, embedding_dim) -> (seq_len, batch_size, embedding_dim).
        e_T = e.transpose(0, 1)
        
        # Pad sequence if masked.
        if m is not None:
            seq_lens = list(map(int, torch.sum(m, dim=1)))
            e_T = torch.nn.utils.rnn.pack_padded_sequence(e_T, seq_lens)
        
        # Pass embeddings through an RNN layer,
        # (seq_len, batch_size, embedding_dim) -> (seq_len, batch_size, hidden_dim).
        hiddens, _ = self.rnn(e_T)
       
        # Pad hiddens if masked.
        if m is not None:
            hiddens, _ = torch.nn.utils.rnn.pad_packed_sequence(hiddens)

        # Permute hiddens,
        # (seq_len, batch_size, hidden_dim) -> (batch_size, hidden_dim, seq_len).
        hiddens = hiddens.permute(1, 2, 0)

        return hiddens


class CnnEncoder(nn.Module):
    """
    Basic CNN encoder module, input embeddings and output hidden states.
    """

    def __init__(self, args):
        """
        args.hidden_dim -- dimension of filters.
        args.kernel_size -- kernel size of the conv1d.
        args.layer_num -- number of CNN layers.
        args.embedding_dim -- dimension of word embeddings.
        """
        super(CnnEncoder, self).__init__()
        self.conv_layers = nn.Sequential()
        for i in range(args.layer_num):
            if i == 0:
                input_dim = args.embedding_dim
            else:
                input_dim = args.hidden_dim
            self.conv_layers.add_module("conv_layer{:d}".format(i),
                                        nn.Conv1d(in_channels=input_dim,
                                                  out_channels=args.hidden_dim,
                                                  kernel_size=args.kernel_size,
                                                  padding=(args.kernel_size-1)/2))
            self.conv_layers.add_module("relu{:d}".format(i), nn.ReLU())


    def forward(self, embeddings):
        """
        Given input embeddings in shape of (batch_size, sequence_length, embedding_dim) generate a 
        sentence embedding tensor (batch_size, sequence_length, hidden_dim).
        Inputs:
            embeddings -- sequence of word embeddings, (batch_size, sequence_length, embedding_dim).
        Outputs:
            hiddens -- sentence embedding tensor, (batch_size, hidden_dim, sequence_length).
        """
        embeddings_ = embeddings.transpose(1, 2)  # (batch_size, embedding_dim, sequence_length)
        hiddens = self.conv_layers(embeddings_)
        print(hiddens)
        return hiddens
