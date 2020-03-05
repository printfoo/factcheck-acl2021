# coding: utf-8


# Catch passed auguments from run script.
import sys, os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--working_dir", type=str, default="models")
parser.add_argument("--data_name", type=str, default="beer_reviews_single_aspect")
parser.add_argument("--embedding_name", type=str, default="")
parser.add_argument("--model_type", type=str, default="RNN")
parser.add_argument("--cell_type", type=str, default="GRU")
parser.add_argument("--hidden_dim", type=int, default=400)
parser.add_argument("--embedding_dim", type=int, default=100)
parser.add_argument("--kernel_size", type=int, default=5)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--fine_tuning", type=bool, default=False)
parser.add_argument("--z_dim", type=int, default=2)
parser.add_argument("--gumbel_temprature", type=float, default=0.1)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--mlp_hidden_dim", type=int, default=50)
parser.add_argument("--dropout_rate", type=float, default=0.4)
parser.add_argument("--use_relative_pos", type=bool, default=True)
parser.add_argument("--max_pos_num", type=int, default=20)
parser.add_argument("--pos_embedding_dim", type=int, default=-1)
parser.add_argument("--fixed_classifier", type=bool, default=True)
parser.add_argument("--fixed_E_anti", type=bool, default=False)
parser.add_argument("--lambda_sparsity", type=float, default=1.0)
parser.add_argument("--lambda_continuity", type=float, default=1.0)
parser.add_argument("--lambda_anti", type=float, default=1.0)
parser.add_argument("--lambda_pos_reward", type=float, default=0.1)
parser.add_argument("--exploration_rate", type=float, default=0.05)
parser.add_argument("--highlight_percentage", type=float, default=0.3)
parser.add_argument("--highlight_count", type=int, default=8)
parser.add_argument("--count_tokens", type=int, default=8)
parser.add_argument("--count_pieces", type=int, default=4)
parser.add_argument("--lambda_acc_gap", type=float, default=1.2)
parser.add_argument("--label_embedding_dim", type=int, default=400)
parser.add_argument("--game_mode", type=str, default="3player")
parser.add_argument("--margin", type=float, default=0.2)
parser.add_argument("--lm_setting", type=str, default="multiple")
parser.add_argument("--lambda_lm", type=float, default=1.0)
parser.add_argument("--ngram", type=int, default=4)
parser.add_argument("--with_lm", type=bool, default=False)
parser.add_argument("--batch_size_ngram_eval", type=int, default=5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--model_prefix", type=str, default="models")
parser.add_argument("--pre_trained_model_prefix", type=str, default="pre_trained_cls.model")
parser.add_argument("--num_iteration", type=int, default=50)
parser.add_argument("--display_iteration", type=int, default=50)
parser.add_argument("--test_iteration", type=int, default=50)
args, extras = parser.parse_known_args()
print("Arguments:", args)
args.extras = extras
args.command = " ".join(["python"] + sys.argv)

# Torch.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # (achtung-gpu) Use the 2nd GPU.

# Set random seeds.
import numpy as np
import random
torch.manual_seed(0)  # Set random seeds.
np.random.seed(0)
random.seed(0)

# Import specified dataset loader.
import importlib
dataset = importlib.import_module("datasets." + args.data_name)

from models.rationale_3players import HardRationale3PlayerClassificationModelForEmnlp
from utils.trainer_utils import evaluate_rationale_model_glue

from collections import deque
from tqdm import tqdm

# Load data.
data_path = os.path.join(args.data_dir, args.data_name)
data = dataset.DataLoader(data_path, score_threshold=0.6, split_ratio=0.1)
args.num_labels = len(data.label_vocab)
print("Data successfully loaded:", data)

# Load or initialize embeddings.
if args.embedding_name:
    embedding_path = os.path.join(args.data_dir, args.embedding_name)
    embeddings = data.initial_embedding(args.embedding_dim, embedding_path)
    print("Embeddings successfully loaded:", embeddings.shape)
else:
    embeddings = data.initial_embedding(args.embedding_dim)
    print("Embeddings successfully initialized:", embeddings.shape)

# Initialize model.
model = HardRationale3PlayerClassificationModelForEmnlp(embeddings, args)
print("Model successfully initialized:", model)


print('training with game mode:', model.game_mode)

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
model.count_tokens = args.count_tokens
model.count_pieces = args.count_pieces
model.init_C_model()
model.fixed_E_anti = args.fixed_E_anti
model.init_optimizers()
model.init_rl_optimizers()
model.init_reward_queue()

old_E_anti_weights = model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()

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
    
    if not args.with_lm:
        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = model.train_one_step(
            batch_x_, batch_y_, z_baseline, batch_m_, with_lm=False)
    else:
        losses, predict, anti_predict, z, z_rewards, continuity_loss, sparsity_loss = model.train_one_step(
            batch_x_, batch_y_, z_baseline, batch_m_, with_lm=True)
    
    z_batch_reward = np.mean(z_rewards.cpu().data.numpy())
    z_history_rewards.append(z_batch_reward)

    # calculate classification accuarcy
    _, y_pred = torch.max(predict, dim=1)
    
    acc = np.float((y_pred == batch_y_).sum().cpu().data) / args.batch_size
    train_accs.append(acc)

    train_losses.append(losses['e_loss'])
    
    if args.fixed_E_anti == True:
        new_E_anti_weights = model.E_anti_model.predictor._parameters['weight'][0].cpu().data.numpy()
        assert (old_E_anti_weights == new_E_anti_weights).all(), 'E anti model changed'
    
    if (i+1) % display_iteration == 0:
        print('sparsity lambda: %.4f'%(model.lambda_sparsity))
        print('highlight percentage: %.4f'%(model.highlight_percentage))
        print('supervised_loss %.4f, sparsity_loss %.4f, continuity_loss %.4f'%(losses['e_loss'], torch.mean(sparsity_loss).cpu().data, torch.mean(continuity_loss).cpu().data))
        if args.with_lm:
            print('lm prob: %.4f'%losses['lm_prob'])
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
            snapshot_path = os.path.join(args.working_dir, args.model_prefix + '.state_dict.bin')
            print('new best dev:', new_best_dev_acc, 'model saved at', snapshot_path)
            torch.save(model.state_dict(), snapshot_path)

        # Eval test set.
        new_best_test_acc = evaluate_rationale_model_glue(model, data, args, test_accs, test_anti_accs, test_cls_accs, best_test_acc, print_train_flag=False, eval_test=True)
        if new_best_test_acc > best_test_acc:
            best_test_acc = new_best_test_acc

