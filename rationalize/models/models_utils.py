# coding: utf-8


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def bao_regularization_loss_batch(z, percentage, mask=None):
    """
    Compute regularization loss, based on a given rationale sequence
    Use Yujia's formulation

    Inputs:
        z -- torch variable, "binary" rationale, (batch_size, sequence_length)
        percentage -- the percentage of words to keep
    Outputs:
        a loss value that contains two parts:
        continuity_loss --  \sum_{i} | z_{i-1} - z_{i} | 
        sparsity_loss -- |mean(z_{i}) - percent|
    """

    # (batch_size,)
    if mask is not None:
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)
    else:
        mask_z = z
        seq_lengths = torch.sum(z - z + 1.0, dim=1)
    
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        
    continuity_loss = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,)
    sparsity_loss = torch.abs(torch.sum(mask_z, dim=-1) / seq_lengths - percentage)  #(batch_size,)

    return continuity_loss, sparsity_loss

