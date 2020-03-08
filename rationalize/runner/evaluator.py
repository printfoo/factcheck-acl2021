# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


def _get_sparsity(z, mask):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1)
    sparsity_count = torch.sum(mask_z, dim=-1)
    sparsity_ratio = sparsity_count / seq_lengths  # (batch_size,).
    return sparsity_ratio, sparsity_count


def _get_continuity(z, mask):
    mask_z = z * mask
    seq_lengths = torch.sum(mask, dim=1) 
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1) 
    continuity_count = torch.sum(torch.abs(mask_z - mask_z_), dim=-1)
    continuity_ratio = continuity_count / seq_lengths  # (batch_size,).
    return continuity_ratio, continuity_count


def evaluate(model, data, args, accs, anti_accs, cls_accs, best_acc, set_name):
    
    model.eval()  # Set model to eval mode.

    correct = 0.0
    anti_correct = 0.0
    cls_correct = 0.0
    sparsity_total = 0.0
    continuity_total = 0.0
    sparsity_count_total = 0.0
    continuity_count_total = 0.0

    instance_count = data.data_sets[set_name].size()
    total = 0
    for start in range(instance_count // args.batch_size):
        
        # Get a batch.
        batch_idx=range(start * args.batch_size, (start + 1) * args.batch_size)
        x_mat, y_vec, x_mask = data.get_batch(set_name, batch_idx=batch_idx, sort=True)

        # Save values to torch tensors.
        batch_x_ = Variable(torch.from_numpy(x_mat))
        batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
        batch_y_ = Variable(torch.from_numpy(y_vec))
        if args.cuda:
            batch_x_ = batch_x_.cuda()
            batch_m_ = batch_m_.cuda()
            batch_y_ = batch_y_.cuda()

        predict, anti_predict, z, neg_log_probs = model(batch_x_, batch_m_)
        cls_predict = predict

        # calculate classification accuarcy
        _, y_pred = torch.max(predict, dim=1)
        _, anti_y_pred = torch.max(anti_predict, dim=1)
        _, y_cls_pred = torch.max(cls_predict, dim=1)

        correct += np.float((y_pred == batch_y_).sum().item())
        anti_correct += np.float((anti_y_pred == batch_y_).sum().item())
        cls_correct += np.float((y_cls_pred == batch_y_).sum().item())
        total += args.batch_size

        if len(z.shape) == 3:
            mask_expand_ = batch_m_.unsqueeze(1).expand(batch_m_.size(0), z.size(1), batch_m_.size(1)).contiguous()  # (batch_size, num_label, length).
            mask_expand_ = mask_expand_.view(-1, batch_m_.size(1))  # (batch_size*num_label, length).
            sparsity_ratio = _get_sparsity(z.view(-1, z.size(2)), mask_expand_)
            sparsity_total += sparsity_ratio.sum().item() / z.size(0)
        else:
            sparsity_ratio, sparsity_count = _get_sparsity(z, batch_m_)
            sparsity_total += sparsity_ratio.sum().item()
            sparsity_count_total += sparsity_count.sum().item()
            continuity_ratio, continuity_count = _get_continuity(z, batch_m_)
            continuity_total += continuity_ratio.sum().item()
            continuity_count_total += continuity_count.sum().item()

        accs.append(correct / total)
        anti_accs.append(anti_correct / total)
        cls_accs.append(cls_correct / total)
        best_acc = correct / total

    print(set_name, "acc:", accs[-1], "best acc:", best_acc, "anti acc:", anti_accs[-1], "cls acc:", cls_accs[-1], "sparsity:", sparsity_total / total, "continuity:", continuity_total / total)
 
    return best_acc
