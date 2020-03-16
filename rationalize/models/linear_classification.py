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
        self.linear = torch.nn.Linear(embeddings.shape[0], args.num_labels)
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)
        self.loss_func = nn.CrossEntropyLoss()


    def forward(self, x):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length).
        Outputs:
            predict -- (batch_size, num_label).
        """
        predict = self.linear(x)
        return predict


    def train_one_step(self, x, label):
        predict = self.forward(x)
        loss = self.loss_func(predict, label) 
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        losses = {"loss": loss.data}
        return losses, predict, None
