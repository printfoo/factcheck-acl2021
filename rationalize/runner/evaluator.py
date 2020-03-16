# coding: utf-8


import torch
from torch.autograd import Variable

import numpy as np
from runner.metrics import get_batch_accuracy


def evaluate(model, data, args, set_name):
    
    model.eval()  # Set model to eval mode.
    accs = []  # Initialize records.

    instance_count = data.data_sets[set_name].size()
    for start in range(instance_count // args.batch_size):

        # Get a batch.
        batch_idx = range(start * args.batch_size, (start + 1) * args.batch_size)
        x, y, m = data.get_batch(set_name, batch_idx=batch_idx, sort=True)

        # Save values to torch tensors.
        x = Variable(torch.from_numpy(x))
        y = Variable(torch.from_numpy(y))
        m = Variable(torch.from_numpy(m)).type(torch.FloatTensor)
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            m = m.cuda()

        # Get predictions.
        forward_tuple = model(x, m)
        if len(forward_tuple) == 1:
            predict = forward_tuple[0]

        # Evaluate classification accuracy.
        _, y_pred = torch.max(predict, dim=1)
        accs.append(get_batch_accuracy(y_pred, y))

    # Average.
    acc = np.mean(accs)
    print(set_name, "acc:", acc)
    return {"acc": acc}
