# coding: utf-8


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(true, pred):
    return accuracy_score(true, pred)


def precision(true, pred):
    if sum(pred) == 0:
        return 0
    return precision_score(true, pred, average="binary")


def recall(true, pred):
    if sum(true) == 0:
        return 0
    return recall_score(true, pred, average="binary")


def f1(true, pred):
    if sum(true) == 0 or sum(pred) == 0:
        return 0
    return f1_score(true, pred, average="binary")


