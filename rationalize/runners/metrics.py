# coding: utf-8


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def percentage(true, pred, mask=None, average=None, invalid=np.nan):
    if mask:
        seq_len = int(sum(mask))
        true = true[:seq_len]
        pred = pred[:seq_len]
    return sum(pred) / len(pred)


def accuracy(true, pred, mask=None, average=None, invalid=np.nan):
    if mask:
        seq_len = int(sum(mask))
        true = true[:seq_len]
        pred = pred[:seq_len]
    return accuracy_score(true, pred)


def precision(true, pred, mask=None, average="macro", invalid=np.nan):
    if mask:
        seq_len = int(sum(mask))
        true = true[:seq_len]
        pred = pred[:seq_len]
    if sum(pred) == 0:
        return invalid
    return precision_score(true, pred, average=average)


def recall(true, pred, mask=None, average="macro", invalid=np.nan):
    if mask:
        seq_len = int(sum(mask))
        true = true[:seq_len]
        pred = pred[:seq_len]
    if sum(true) == 0:
        return invalid
    return recall_score(true, pred, average=average)


def f1(true, pred, mask=None, average="macro", invalid=np.nan):
    if mask:
        seq_len = int(sum(mask))
        true = true[:seq_len]
        pred = pred[:seq_len]
    if sum(true) == 0 or sum(pred) == 0:
        return invalid
    return f1_score(true, pred, average=average)


