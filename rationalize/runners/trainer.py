# coding: utf-8

import torch
from torch.autograd import Variable

import numpy as np
import random, os, json
from tqdm import tqdm

from runners.evaluator import evaluate
from runners.metrics import accuracy


def train(model, data, args):
   
    # Use GPU.
    if args.cuda:
        model.cuda()
    print("Using GPU:", torch.cuda.current_device())

    # Initialize records.
    accs = {"name": "accuracy", "train": [], "dev": [], "test": []}
    tmp_acc = 0.0

    # Start training iterations.
    for i in tqdm(range(args.num_iteration + 1)):

        model.train()  # Set model to train mode.
        x, y, _, m = data.get_train_batch(batch_size=args.batch_size,
                                          sort=True)  # Sample a batch.

        # Save values to torch tensors.
        x = Variable(torch.from_numpy(x))
        y = Variable(torch.from_numpy(y))
        m = Variable(torch.from_numpy(m)).float()
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            m = m.cuda()

        # Train one step.
        losses, predict, z = model.train_one_step(x, y, m)

        # Evaluate classification accuracy.
        _, y_pred = torch.max(predict, dim=1)
        tmp_acc += accuracy(y.tolist(), y_pred.tolist())

        # Display every args.display_iteration.
        if args.display_iteration and i % args.display_iteration == 0:
            y_ = y[2]
            pred_ = y_pred.data[2]
            x_ = x[2,:]
            z_ = z.data[2,:]
            z_b = torch.zeros_like(z)
            z_b_ = z_b.data[2,:]
            print("gold label:", data.idx2label[y_.item()], "pred label:", data.idx2label[pred_.item()])
            data.display_example(x_, z_)

        # Eval every args.eval_iteration.
        if args.eval_iteration and i % args.eval_iteration == 0:

            # Eval dev set.
            metrics, _ = evaluate(model, data, args, "dev")
            accs["dev"].append(metrics["accuracy"])

            # Eval test set.
            metrics, _ = evaluate(model, data, args, "test")
            accs["test"].append(metrics["accuracy"])

            # Adds train set metrics.
            accs["train"].append(tmp_acc / args.eval_iteration)
            tmp_acc = 0.0

            # Save checkpoint.
            snapshot_path = os.path.join(args.working_dir, "i_{:05d}.pt".format(i))
            torch.save(model, snapshot_path)


    record_path = os.path.join(args.working_dir, accs["name"] + ".json")
    with open(record_path, "w") as f:
        f.write(json.dumps(accs))
