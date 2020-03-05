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
from trainer.train_model import train_model


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

train_model(model, data, args)

