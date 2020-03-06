# coding: utf-8


# Catch passed auguments from run script.
import sys, os
import argparse
parser = argparse.ArgumentParser()

# Mode arguments.
parser.add_argument("--mode", type=str, default="train",
                    help="Run mode, train or eval.")

# Environment arguments.
parser.add_argument("--data_dir", type=str, default="data",
                    help="Data folder name.")
parser.add_argument("--working_dir", type=str, default="models",
                    help="Model folder name.")
parser.add_argument("--data_name", type=str, default="beer_reviews_single_aspect",
                    help="Dataset name.")
parser.add_argument("--embedding_name", type=str, default="",
                    help="Embedding name.")
parser.add_argument("--random_seed", type=int, default=0,
                    help="Random seed.")
parser.add_argument("--cuda", type=bool, default=True,
                    help="If use CUDA GPU.")
parser.add_argument("--gpu_id", type=str, default="1",
                    help="ID of the GPU chip to use.")

# Data arguments.
parser.add_argument("--score_threshold", type=float, default=0.6,
                    help="Threshold for positive/negative labels.")
parser.add_argument("--split_ratio", type=float, default=0.1,
                    help="Split ratio for train/dev sets.")
parser.add_argument("--aspect", type=int, default=1,
                    help="Aspect of beer, 0-3 for apperance, aroma, palate, taste.")
parser.add_argument("--freq_threshold", type=int, default=1,
                    help="Minimum frequency for vocabulary.")
parser.add_argument("--truncate_num", type=int, default=300,
                    help="Maximum number of tokens to truncate.")

# Model arguments.,
parser.add_argument("--model_type", type=str, default="RNN")
parser.add_argument("--cell_type", type=str, default="GRU")
parser.add_argument("--hidden_dim", type=int, default=400)
parser.add_argument("--embedding_dim", type=int, default=100)
parser.add_argument("--kernel_size", type=int, default=5)
parser.add_argument("--layer_num", type=int, default=1)
parser.add_argument("--fine_tuning", type=bool, default=False)
parser.add_argument("--z_dim", type=int, default=2)
parser.add_argument("--gumbel_temprature", type=float, default=0.1)
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

# Training arguments.
parser.add_argument("--num_iteration", type=int, default=50,
                    help="Number of iterations to train.")
parser.add_argument("--display_iteration", type=int, default=50,
                    help="Number of iterations to display results.")
parser.add_argument("--test_iteration", type=int, default=50,
                    help="Number of iterations to test.")

# Parse arguments.
args, extras = parser.parse_known_args()
args.extras = extras
args.command = " ".join(["python"] + sys.argv)
print("Command with argumanets:", args.command)

# Load data and embeddings.
import importlib
dataset = importlib.import_module("datasets." + args.data_name)
data_path = os.path.join(args.data_dir, args.data_name)
data = dataset.DataLoader(data_path, args)  # Load data.
args.num_labels = len(data.label_vocab)  # Number of labels.
embedding_path = os.path.join(args.data_dir, args.embedding_name)
embeddings = data.initial_embedding(args.embedding_dim, embedding_path)  # Load embeddings.
print("Data and embeddings successfully loaded:", data, embeddings.shape)

# Initialize model.
from models.rationale_3players import HardRationale3PlayerClassificationModelForEmnlp
model = HardRationale3PlayerClassificationModelForEmnlp(embeddings, args)
print("Model successfully initialized:", model)

# Train model.
from trainer.train_model import train_model
train_model(model, data, args)
print("Model successfully trained.")
