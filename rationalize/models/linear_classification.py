# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LinearClassification(nn.Module):
    """
    LinearClassification model.
    """

    def __init__(self, embeddings, args):
        super(LinearClassification, self).__init__()
        self.args = args
        self.use_cuda = args.cuda

        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings)
        self.linear = torch.nn.Linear(self.embedding_dim, args.num_labels)
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)
        self.loss_func = nn.CrossEntropyLoss()


    def _create_embed_layer(self, embeddings):
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = bool(self.args.fine_tuning)
        return embed_layer


    def forward(self, x, m):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length).
        Outputs:
            predict -- (batch_size, num_label).
        """
        word_embeddings = self.embed_layer(x)  # (batch_size, seq_len, embedding_dim).
        doc_embedding = word_embeddings.sum(dim=1)  # (batch_size, embedding_dim).
        predict = self.linear(doc_embedding)
        return predict, None, None, None


    def train_one_step(self, x, y, m=None):
        """
        Inputs:
            x -- Variable() of input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            y -- Variable() of input x, shape (batch_size,),
                 only one element per instance 0-|label| pointing to a label.
            m -- Variable() of input x, shape (batch_size, seq_len).
                 each element in the seq_len is of 0/1 selecting a token or not.
                 (Not used in this model.)
        Outputs:
            losses -- a dict storing some losses, only one loss in this model.
            predict -- prediction of the label, shape (batch_size,).
            z -- the predicted rationale. (Not used in this model.)
        """
        predict, _, _, _ = self.forward(x, m)
        loss = self.loss_func(predict, y)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        losses = {"loss": loss.data}
        return losses, predict, None
