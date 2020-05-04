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
    metrics_records = {"dev": [], "test": []}

    # Start training iterations.
    for i in tqdm(range(args.num_iteration + 1)):

        model.train()  # Set model to train mode.
        samples = data.get_train_batch(args.batch_size, sort=True)  # Sample a batch.
        x, y, m, r, s, d = samples

        # Save values to torch tensors.
        x = Variable(torch.from_numpy(x))
        y = Variable(torch.from_numpy(y))
        m = Variable(torch.from_numpy(m)).float()
        r = Variable(torch.from_numpy(r)).float()
        s = Variable(torch.from_numpy(s)).float()
        d = Variable(torch.from_numpy(d)).float()
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
            m = m.cuda()
            r = r.cuda()
            s = s.cuda()
            d = d.cuda()

        # Train one step.
        losses, predict, z = model.train_one_step(x, y, m, r, s, d)

        # Display every args.display_iteration.
        if args.display_iteration and i % args.display_iteration == 0:
            _, y_pred = torch.max(predict, dim=1)
            y_ = y[2]
            y_pred = torch.max(predict, dim=1)[1]
            pred_ = y_pred.data[2]
            x_ = x[2,:]
            z_ = z.data[2,:]
            z_b = torch.zeros_like(z)
            z_b_ = z_b.data[2,:]
            data.display_example(x_, z_)
            print("gold label:", data.idx2label[y_.item()])
            print("pred label:", data.idx2label[pred_.item()])

        # Eval every args.eval_iteration.
        if args.eval_iteration and i % args.eval_iteration == 0:

            # Eval dev set.
            metrics = evaluate(model, data, args, "dev")
            print(metrics)
            metrics_records["dev"].append(metrics)

            # Eval test set.
            metrics = evaluate(model, data, args, "test")
            print(metrics)
            metrics_records["test"].append(metrics)

            # Save checkpoint.
            snapshot_path = os.path.join(args.working_dir, "i_{:05d}.pt".format(i))
            torch.save(model, snapshot_path)
            print("[Checkpoint saved.]")

    record_path = os.path.join(args.working_dir, "record.json")
    with open(record_path, "w") as f:
        f.write(json.dumps(metrics_records))
