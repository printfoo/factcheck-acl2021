# coding: utf-8


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(true, pred, average=None):
    return accuracy_score(true, pred)


def precision(true, pred, average="macro"):
    if sum(pred) == 0:
        return np.nan
    return precision_score(true, pred, average=average)


def recall(true, pred, average="macro"):
    if sum(true) == 0:
        return np.nan
    return recall_score(true, pred, average=average)


def f1(true, pred, average="macro"):
    if sum(true) == 0 or sum(pred) == 0:
        return np.nan
    return f1_score(true, pred, average=average)


