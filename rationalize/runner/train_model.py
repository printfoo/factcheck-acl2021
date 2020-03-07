# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import random, os

from runner.train_utils import evaluate_rationale_model_glue

from collections import deque
from tqdm import tqdm

def train_model(model, data, args):
    torch.manual_seed(args.random_seed)  # Set random seeds.
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id # (achtung-gpu) Use the 2nd GPU.
    #print('training with game mode:', model.game_mode)

    train_losses = []
    train_accs = []
    dev_accs = [0.0]
    dev_anti_accs = [0.0]
    dev_cls_accs = [0.0]
    test_accs = [0.0]
    test_anti_accs = [0.0]
    test_cls_accs = [0.0]
    best_dev_acc = 0.0
    best_test_acc = 0.0
    num_iteration = args.num_iteration
    display_iteration = args.display_iteration
    test_iteration = args.test_iteration

    eval_accs = [0.0]
    eval_anti_accs = [0.0]

    queue_length = 200
    z_history_rewards = deque(maxlen=queue_length)
    z_history_rewards.append(0.)

    if args.cuda:
        model.cuda()

    #old_E_anti_weights = model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()

    for i in tqdm(range(num_iteration)):
        model.train()

        # sample a batch of data
        x_mat, y_vec, x_mask = data.get_train_batch(batch_size=args.batch_size, sort=True)

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

        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = model.train_one_step(
            batch_x_, batch_y_, z_baseline, batch_m_)#, with_lm=args.with_lm)

        z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
        z_history_rewards.append(z_batch_reward)

        # calculate classification accuarcy
        _, y_pred = torch.max(predict, dim=1)

        acc = np.float((y_pred == batch_y_).sum().cpu().data) / args.batch_size
        train_accs.append(acc)

        train_losses.append(losses['e_loss'])

        # if args.fixed_E_anti == True:
        #    new_E_anti_weights = model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()
        #    assert (old_E_anti_weights == new_E_anti_weights).all(), 'E anti model changed'

        if (i+1) % display_iteration == 0:
            print('sparsity lambda: %.4f'%(model.lambda_sparsity))
            print('highlight percentage: %.4f'%(model.highlight_percentage))
            print('supervised_loss %.4f, sparsity_loss %.4f, continuity_loss %.4f'%(losses['e_loss'], torch.mean(sparsity_loss).cpu().data, torch.mean(continuity_loss).cpu().data))
            # if args.with_lm:
            #    print('lm prob: %.4f'%losses['lm_prob'])
            y_ = y_vec[2]
            pred_ = y_pred.data[2]
            x_ = x_mat[2,:]
            if len(z.shape) == 3:
                z_ = z.cpu().data[2,pred_,:]
            else:
                z_ = z.cpu().data[2,:]

            z_b = torch.zeros_like(z)
            z_b_ = z_b.cpu().data[2,:]
            print('gold label:', data.idx2label[y_], 'pred label:', data.idx2label[pred_.item()])
            data.display_example(x_, z_)

        if (i+1) % test_iteration == 0:

            # Eval dev set.
            new_best_dev_acc = evaluate_rationale_model_glue(model, data, args, dev_accs, dev_anti_accs, dev_cls_accs, best_dev_acc, print_train_flag=False)
            if new_best_dev_acc > best_dev_acc:
                best_dev_acc = new_best_dev_acc
                snapshot_path = os.path.join(args.working_dir, "trained.ckpt")
                print('new best dev:', new_best_dev_acc, 'model saved at', snapshot_path)
                torch.save(model.state_dict(), snapshot_path)

            # Eval test set.
            new_best_test_acc = evaluate_rationale_model_glue(model, data, args, test_accs, test_anti_accs, test_cls_accs, best_test_acc, print_train_flag=False, eval_test=True)
            if new_best_test_acc > best_test_acc:
                best_test_acc = new_best_test_acc
