# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.encoder import RnnEncoder, CnnEncoder, TrmEncoder


class Classifier(nn.Module):
    """
    Classifier module, input sequence and binary mask, output label.
    """

    def __init__(self, args):
        """
        Inputs:
            args.num_labels -- number of labels.
            args.hidden_dim -- dimension of hidden states.
            args.model_type -- type of model, RNN/CNN/TRM.
            args.layer_num -- number of layers.
            args.cell_type -- type of cell GRU or LSTM (RNN only).
            args.kernel_size -- kernel size of the conv1d (CNN only).
            args.head_num -- number of heads for multi head attention (TRM only).
            args.embedding_dim -- dimension of word embeddings.
        """
        super(Classifier, self).__init__()
        self.NEG_INF = -1.0e6
        encoders = {"RNN": RnnEncoder, "CNN": CnnEncoder, "TRM": TrmEncoder}
        self.encoder = encoders[args.model_type](args)
        self.predictor = nn.Linear(args.hidden_dim, args.num_labels)


    def forward(self, e, z, m):
        """
        Inputs:
            e -- input sequence with embeddings, shape (batch_size, seq_len, embedding_dim),
                 each element in the seq_len is a word embedding of embedding_dim.
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            m -- mask of the input sequence, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            predict -- prediction score of classifier, shape (batch_size, |label|),
                       each element at i is a predicted probability for label[i].
        """

        # Get rationales by masking input sequence with rationale selection z.
        rationales = e * z.unsqueeze(-1)

        # Pass rationales through an encoder and get hidden states,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, hidden_dim, seq_len).
        hiddens = self.encoder(rationales, m)

        # Get max hidden of a sequence from hiddens,
        # Here hiddens are masked by rationale selection z again (m * z),
        # (batch_size, hidden_dim, seq_len) -> (batch_size, hidden_dim)
        max_hidden = torch.max(hiddens + (1 - m * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]

        # Pass max hidden to an output linear layer and get prediction,
        # (batch_size, hidden_dim) -> (batch_size, |label|).
        predict = self.predictor(max_hidden)

        return predict
