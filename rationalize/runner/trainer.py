# coding: utf-8

import torch
from torch.autograd import Variable

import numpy as np
import random, os, json
from collections import deque
from tqdm import tqdm

from runner.evaluator import evaluate
from runner.metrics import get_batch_accuracy, get_batch_sparsity, get_batch_continuity


def train(model, data, args):
    
    # Set GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.cuda:
        model.cuda()

    # Initialize reward queue.
    z_history_rewards = deque(maxlen=200)
    z_history_rewards.append(0.)

    # Initialize records.
    accs = {"name": "accuracy", "train": [], "dev": [], "test": []}
    anti_accs = {"name": "anti-accuracy", "train": [], "dev": [], "test": []}
    sparsities = {"name": "sparsity", "train": [], "dev": [], "test": []}
    continuities = {"name": "continuity", "train": [], "dev": [], "test": []}

    best_dev_acc = 0.0
    best_test_acc = 0.0
    tmp_acc = 0.0
    tmp_anti_acc = 0.0
    tmp_sparsity = 0.0
    tmp_continuity = 0.0

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
        z_baseline = Variable(torch.FloatTensor([float(np.mean(z_history_rewards))]))
        if args.cuda:
            z_baseline = z_baseline.cuda()

        # Train one step.
        loss_tuple = model.train_one_step(batch_x_, batch_y_, z_baseline, batch_m_)

        # Update losses.
        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = loss_tuple
        z_batch_reward = torch.mean(z_rewards).item()
        z_history_rewards.append(z_batch_reward)

        # Evaluate classification accuracy.
        _, y_pred = torch.max(predict, dim=1)
        _, anti_y_pred = torch.max(anti_predict, dim=1)
        tmp_acc += get_batch_accuracy(y_pred, batch_y_)
        tmp_anti_acc += get_batch_accuracy(anti_y_pred, batch_y_)

        # Evaluate sparsity and continuity measures.
        tmp_sparsity += get_batch_sparsity(z, batch_m_)
        tmp_continuity += get_batch_continuity(z, batch_m_)

        # Display every args.display_iteration.
        if i % args.display_iteration == 0:
            print("supervised_loss %.4f, sparsity_loss %.4f, continuity_loss %.4f" %
                  (losses["e_loss"], torch.mean(sparsity_loss).data, torch.mean(continuity_loss).data))
            y_ = y_vec[2]
            pred_ = y_pred.data[2]
            x_ = x_mat[2,:]
            z_ = z.data[2,:]
            z_b = torch.zeros_like(z)
            z_b_ = z_b.data[2,:]
            print("gold label:", data.idx2label[y_], "pred label:", data.idx2label[pred_.item()])
            data.display_example(x_, z_)

        if i % args.eval_iteration == 0:

            # Eval dev set.
            dev_acc, dev_anti_acc, dev_sparsity, dev_continuity = evaluate(model, data, args, "dev")
            accs["dev"].append(dev_acc)
            anti_accs["dev"].append(dev_anti_acc)
            sparsities["dev"].append(dev_sparsity)
            continuities["dev"].append(dev_continuity)
            if dev_acc > best_dev_acc:  # If historically best on dev set.
                best_dev_acc = dev_acc
                snapshot_path = os.path.join(args.working_dir, "trained.ckpt")
                print("new best dev:", dev_acc, "model saved at", snapshot_path)
                torch.save(model.state_dict(), snapshot_path)

            # Eval test set.
            test_acc, test_anti_acc, test_sparsity, test_continuity = evaluate(model, data, args, "test")
            accs["test"].append(test_acc)
            anti_accs["test"].append(test_anti_acc)
            sparsities["test"].append(test_sparsity)
            continuities["test"].append(test_continuity)
            if test_acc > best_test_acc:  # If historically best on test set.
                best_test_acc = test_acc

            # Adds train set metrics.
            accs["train"].append(tmp_acc / args.eval_iteration)
            anti_accs["train"].append(tmp_anti_acc / args.eval_iteration)
            sparsities["train"].append(tmp_sparsity / args.eval_iteration)
            continuities["train"].append(tmp_continuity / args.eval_iteration)
            tmp_acc = 0.0
            tmp_anti_acc = 0.0
            tmp_sparsity = 0.0
            tmp_continuity = 0.0

            # Save checkpoint.
            snapshot_path = os.path.join(args.working_dir, "i_%s.ckpt" % i)
            torch.save(model.state_dict(), snapshot_path)

    for metric in [accs, anti_accs, sparsities, continuities]:
        record_path = os.path.join(args.working_dir, metric["name"] + ".json")
        with open(record_path, "w") as f:
            f.write(json.dumps(metric))
        print("Training record saved for:", metric["name"])
    return accs, anti_accs, sparsities, continuities
