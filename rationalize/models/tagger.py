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

    def __init__(self, args):
        """
        Inputs:
            args.z_dim -- dimension of rationale, always 2.
            args.hidden_dim -- dimension of hidden states.
            args.layer_num -- number of RNN layers.
            args.cell_type -- type of RNN cells, "GRU" or "LSTM".
            args.embedding_dim -- dimension of word embeddings.
        """
        super(Tagger, self).__init__()
        self.NEG_INF = -1.0e6
        self.encode_layer = RnnModel(args, args.embedding_dim)
        self.output_layer = nn.Linear(args.hidden_dim, args.z_dim)


    def _binarize_probs(self, z_probs):
        """
        Binarize a probability distribution.
        Input:
            z_prob_ -- probability of selecting rationale, shape (batch_size, seq_len, 2),
                       each element is a 0-1 probability of selecting a token or not.
        Output:
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            neg_log_probs -- negative log probability, shape (batch_size, seq_len).
        """

        # Reshape z_probs by concatenating all batches,
        # (batch_size, seq_len, 2) -> (batch_size * seq_len, 2)
        z_probs_all = z_probs.view(-1, 2)

        # Create a categorical distribution parameterized by concatenated probs.
        sampler = torch.distributions.Categorical(z_probs_all)

        if self.training:  # If train, sample rationale from the distribution.
            z_all = sampler.sample()  # (batch_size * seq_len).
        else:  # If eval, use max prob as rationale.
            z_all = torch.max(z_probs_all, dim=-1)[1]  # (batch_size * seq_len).
        neg_log_probs_all = -sampler.log_prob(z_all)  # (batch_size * seq_len).

        # Recover concatenated rationales to each batch,
        # (batch_size * seq_len) -> (batch_size, seq_len).
        z = z_all.view(z_probs.size(0), z_probs.size(1))
        neg_log_probs = neg_log_probs_all.view(z_probs.size(0), z_probs.size(1))

        return z.float(), neg_log_probs


    def forward(self, e, m):
        """
        Inputs:
            e -- Input sequence with embeddings, shape (batch_size, seq_len, embedding_dim),
                 each element in the seq_len is a word embedding of embedding_dim.
            m -- Mask of the input sequence, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            neg_log_probs -- negative log probability, shape (batch_size, seq_len).
        """

        # Pass embeddings through an RNN module and get hidden states,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim).
        hiddens = self.encode_layer(e, m).transpose(1, 2).contiguous()

        # Pass hidden states to an output linear layer and get rationale scores,
        # (batch_size, seq_len, hidden_dim) -> (batch_size, seq_len, 2).
        z_scores = self.output_layer(hiddens)

        # Replace (batch_size, seq_len, 1) with -inf.
        z_scores[:, :, 1] = z_scores[:, :, 1] + (1 - m) * self.NEG_INF

        # Run a softmax for valid probs (batch_size, seq_len, 0) + (batch_size, seq_len, 1) = 1.
        z_probs = F.softmax(z_scores, dim=-1)

        # Generate rationale and negative log probs,
        # (batch_size, seq_len, 2) -> (batch_size, seq_len)
        z, neg_log_probs = self._binarize_probs(z_probs)

        return z, neg_log_probs
