# coding: utf-8


# Catch passed auguments from run script.
import sys, os, json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train",
                    help="Run mode, train or eval.")
parser.add_argument("--data_dir", type=str, default="data",
                    help="Data folder name.")
parser.add_argument("--data_name", type=str, default="personal_attacks",
                    help="Dataset name.")
parser.add_argument("--config_name", type=str, default="best",
                    help="Dataset name.")
parser.add_argument("--output_dir", type=str, default="output",
                    help="Output folder name.")
parser.add_argument("--random_seed", type=str, default=0,
                    help="Random seed")
args, _ = parser.parse_known_args()
args.data_path = os.path.join(args.data_dir, args.data_name)
args.config_dir = os.path.join(args.data_path, args.config_name + ".config")

if args.mode == "train":

    # Set random seeds.
    import torch
    import numpy as np
    import random
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    
    # Train arguments.
    from utils.checkpointer import init_ckpt
    args.config_dir = os.path.join(args.data_dir, args.data_name, args.config_name + ".config")
    with open(args.config_dir, "r") as f:
        config = json.load(f)
    train_args = argparse.Namespace(**config)
    train_args.embedding_dir = os.path.join(args.data_dir, train_args.embedding_name,
                                            train_args.embedding_name + ".6B.%sd.txt" % train_args.embedding_dim)
    train_args.working_dir = os.path.join(args.data_path, args.config_name + ".ckpt")
    init_ckpt(train_args.working_dir)

    # Load data and embeddings.
    from datasets.dataset_loader import SentenceClassification
    data = SentenceClassification(args.data_path, train_args)  # Load data.
    train_args.num_labels = len(data.label_vocab)  # Number of labels.
    embeddings = data.initial_embedding(train_args.embedding_dim, train_args.embedding_dir)  # Load embeddings.
    print("Data and embeddings successfully loaded:", data, embeddings.shape)

    # Initialize model.
    from models.rationale_3player_classification import Rationale3PlayerClassification
    model = Rationale3PlayerClassification(embeddings, train_args)
    print("Model successfully initialized:", model)

    # Train model.
    from runner.trainer import train
    train(model, data, train_args)
    print("Model successfully trained.")


elif args.mode == "purge":

    # Perge all checkpoints.
    from utils.checkpointer import purge
    purge(os.path.join(args.data_path, args.config_name + ".ckpt"))
    print("All checkpoints purged.")


else:
    exit("Wrong mode.")
