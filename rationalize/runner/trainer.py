# coding: utf-8

import torch
from torch.autograd import Variable

import numpy as np
import random, os, json
from tqdm import tqdm

from runner.evaluator import evaluate
from runner.metrics import get_batch_accuracy


def train(model, data, args):
    
    # Set GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.cuda:
        model.cuda()

    # Initialize records.
    accs = {"name": "accuracy", "train": [], "dev": [], "test": []}
    tmp_acc = 0.0

    # Start training iterations.
    for i in tqdm(range(args.num_iteration + 1)):

        model.train()  # Set model to train mode.
        x_mat, y_vec, x_mask = data.get_train_batch(batch_size=args.batch_size, sort=True)  # Sample a batch.

        # Save values to torch tensors.
        batch_x_ = Variable(torch.from_numpy(x_mat))
        batch_m_ = Variable(torch.from_numpy(x_mask)).type(torch.FloatTensor)
        batch_y_ = Variable(torch.from_numpy(y_vec))
        if args.cuda:
            batch_x_ = batch_x_.cuda()
            batch_m_ = batch_m_.cuda()
            batch_y_ = batch_y_.cuda()

        # Train one step.
        losses, predict, z = model.train_one_step(batch_x_, batch_y_, batch_m_)

        # Evaluate classification accuracy.
        _, y_pred = torch.max(predict, dim=1)
        tmp_acc += get_batch_accuracy(y_pred, batch_y_)

        # Display every args.display_iteration.
        if args.display_iteration and i % args.display_iteration == 0:
            y_ = y_vec[2]
            pred_ = y_pred.data[2]
            x_ = x_mat[2,:]
            z_ = z.data[2,:]
            z_b = torch.zeros_like(z)
            z_b_ = z_b.data[2,:]
            print("gold label:", data.idx2label[y_], "pred label:", data.idx2label[pred_.item()])
            data.display_example(x_, z_)

        # Eval every args.eval_iteration.
        if args.eval_iteration and i % args.eval_iteration == 0:

            # Eval dev set.
            metrics = evaluate(model, data, args, "dev")
            accs["dev"].append(metrics["acc"])

            # Eval test set.
            metrics = evaluate(model, data, args, "test")
            accs["test"].append(metrics["acc"])

            # Adds train set metrics.
            accs["train"].append(tmp_acc / args.eval_iteration)
            tmp_acc = 0.0

            # Save checkpoint.
            snapshot_path = os.path.join(args.working_dir, "i_%s.ckpt" % i)
            torch.save(model.state_dict(), snapshot_path)


    record_path = os.path.join(args.working_dir, accs["name"] + ".json")
    with open(record_path, "w") as f:
        f.write(json.dumps(accs))
