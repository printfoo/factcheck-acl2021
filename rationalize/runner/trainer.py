# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random, os
from collections import deque
from tqdm import tqdm

from runner.evaluator import evaluate


def train(model, data, args):

    # Set random seeds.
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    # Set GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.cuda:
        model.cuda()

    # Initialize reward queue.
    z_history_rewards = deque(maxlen=200)
    z_history_rewards.append(0.)

    # Initialize record for accuracy and losses.
    train_losses = []
    train_accs = []
    best_dev_acc = 0.0
    best_test_acc = 0.0

    # Start training iterations.
    for i in tqdm(range(args.num_iteration)):

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
        z_baseline = Variable(torch.FloatTensor([float(np.mean(z_history_rewards))]))
        if args.cuda:
            z_baseline = z_baseline.cuda()

        # Train one step.
        loss_tuple = model.train_one_step(batch_x_, batch_y_, z_baseline, batch_m_)

        # Update losses.
        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = loss_tuple
        z_batch_reward = torch.mean(z_rewards).item()
        z_history_rewards.append(z_batch_reward)

        # Evaluate classification accuarcy.
        _, y_pred = torch.max(predict, dim=1)
        acc = np.float((y_pred == batch_y_).sum().data) / args.batch_size
        train_accs.append(acc)
        train_losses.append(losses["e_loss"])

        # Display every args.display_iteration.
        if (i+1) % args.display_iteration == 0:
            print("supervised_loss %.4f, sparsity_loss %.4f, continuity_loss %.4f" %
                  (losses["e_loss"], torch.mean(sparsity_loss).data, torch.mean(continuity_loss).data))
            y_ = y_vec[2]
            pred_ = y_pred.data[2]
            x_ = x_mat[2,:]
            if len(z.shape) == 3:
                z_ = z.data[2,pred_,:]
            else:
                z_ = z.data[2,:]
            z_b = torch.zeros_like(z)
            z_b_ = z_b.data[2,:]
            print("gold label:", data.idx2label[y_], "pred label:", data.idx2label[pred_.item()])
            data.display_example(x_, z_)

        if (i+1) % args.test_iteration == 0:

            # Eval dev set.
            new_dev_acc, new_dev_anti_acc, _, _ = evaluate(model, data, args, "dev")
            if new_dev_acc > best_dev_acc:  # If historically best on dev set.
                best_dev_acc = new_dev_acc
                snapshot_path = os.path.join(args.working_dir, "trained.ckpt")
                print('new best dev:', new_dev_acc, 'model saved at', snapshot_path)
                torch.save(model.state_dict(), snapshot_path)

            # Eval test set.
            new_test_acc, new_test_anti_acc, _, _ = evaluate(model, data, args, "test")
            if new_test_acc > best_test_acc:  # If historically best on test set.
                best_test_acc = new_test_acc
