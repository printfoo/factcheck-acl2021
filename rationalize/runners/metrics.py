# coding: utf-8


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(true, pred, average=None, invalid=np.nan):
    return accuracy_score(true, pred)


def percentage(true, pred, average=None, invalid=np.nan):
    return sum(pred) / len(pred)


def precision(true, pred, average="macro", invalid=np.nan):
    if sum(pred) == 0:
        return invalid
    return precision_score(true, pred, average=average)


def recall(true, pred, average="macro", invalid=np.nan):
    if sum(true) == 0:
        return invalid
    return recall_score(true, pred, average=average)


def f1(true, pred, average="macro", invalid=np.nan):
    if sum(true) == 0 or sum(pred) == 0:
        return invalid
    return f1_score(true, pred, average=average)


