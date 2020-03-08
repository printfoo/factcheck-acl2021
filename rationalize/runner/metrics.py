# coding: utf-8


import torch


def get_batch_accuracy(y_pred, y):
    acc = y_pred == y  # (batch_size,).
    return acc.sum().item() / acc.shape[0]


def get_batch_sparsity(z, mask):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    sparsity_count = torch.sum(mask_z, dim=-1)
    sparsity_ratio = sparsity_count / seq_lengths  # (batch_size,).
    return sparsity_ratio.sum().item() / sparsity_ratio.shape[0]


def get_batch_continuity(z, mask):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1) 
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1) 
    continuity_count = torch.sum(torch.abs(mask_z - mask_z_), dim=-1)
    continuity_ratio = continuity_count / seq_lengths  # (batch_size,).
    return continuity_ratio.sum().item() / continuity_ratio.shape[0]
