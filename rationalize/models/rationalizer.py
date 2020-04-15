# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from collections import deque

from models.tagger import Tagger
from models.classifier import Classifier


class Rationalizer(nn.Module):
    """
    Rationalizer model.
    Using model.Tagger and model.Classifier modules.
    """

    def __init__(self, embeddings, args):
        super(Rationalizer, self).__init__()

        # Initialize parameters.
        self.use_cuda = args.cuda
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti
        self.rationale_len = args.rationale_len
        self.rationale_num = args.rationale_num
        self.vocab_size, self.embedding_dim = embeddings.shape

        # Initialize modules.
        self.embed_layer = self._create_embed_layer(embeddings, bool(args.fine_tuning))
        self.classifier = Classifier(args)
        self.anti_classifier = Classifier(args)
        self.tagger = Tagger(args)
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

        # Initialize optimizers.
        p_grad = lambda module: filter(lambda _: _.requires_grad, module.parameters())
        self.opt_classifier = torch.optim.Adam(p_grad(self.classifier), lr=args.lr)
        self.opt_anti_classifier = torch.optim.Adam(p_grad(self.anti_classifier), lr=args.lr)
        self.opt_tagger = torch.optim.Adam(p_grad(self.tagger), lr=args.lr*0.1)

        # Initialize reward queue for reinforce-style loss.
        self.z_history_rewards = deque(maxlen=200)
        self.z_history_rewards.append(0.)


    def _create_embed_layer(self, embeddings, fine_tuning=False):
        """
        Create a lookup layer for embeddings.
        Input:
            embeddings -- embeddings of tokens, shape (|vocab|, embedding_dim). 
        Output:
            embed_layer -- a lookup layer for embeddings,
                           inputs token' ID and returns token's embedding.
        """
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = fine_tuning
        return embed_layer


    def forward(self, x, m):
        """
        Forward model from x and m and get rationales, predictions, etc.
        Inputs:
            x -- Variable() of input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            m -- Variable() of mask m, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            predict -- prediction score of classifier, shape (batch_size, |label|),
                       each element at i is a predicted probability for label[i].
            anti_predict -- prediction score of anti classifier, shape (batch_size, |label|),
                            each element at i is a predicted probability for label[i].
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            neg_log_probs -- negative log probability, shape (batch_size, seq_len).
        """

        # Lookup embeddings of each token,
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim).
        embeddings = self.embed_layer(x)

        # Rationale and negative log probs,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len).
        z, neg_log_probs = self.tagger(embeddings, m)

        # Prediction of classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        predict = self.classifier(embeddings, z, m)
        
        # Prediction of anti classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        anti_predict = self.anti_classifier(embeddings, 1 - z, m)

        return predict, anti_predict, z, neg_log_probs
    

    def _get_tagger_loss(self, accuracy_c, accuracy_anti_c, loss_continuity, loss_sparsity, z, neg_log_probs, m):


        # Baseline reward for reinforce-style learning, average of history.
        rl_base = Variable(torch.tensor(np.mean(self.z_history_rewards)))
        if self.use_cuda:
            rl_base = rl_base.cuda()

        rewards = accuracy_c - accuracy_anti_c * self.lambda_anti - loss_continuity * self.lambda_continuity - loss_sparsity * self.lambda_sparsity  # batch RL reward.

        advantages = rewards - rl_base  # (batch_size,).
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()

        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)  # (batch_size, q_length).
        loss_rl = torch.sum(neg_log_probs * advantages_expand_ * m)

        # Update reinforce-style loss of this batch to the history reward queue.
        z_batch_reward = torch.mean(rewards).item()
        self.z_history_rewards.append(z_batch_reward)

        return loss_rl


    def _get_regularization_loss(self, z, rationale_len, rationale_num, mask=None):
        """
        Compute regularization loss, based on a given rationale sequence.
        Inputs:
            z -- torch variable, "binary" rationale, (batch_size, sequence_length).
            rationale_len -- suggested upper bound of total tokens of all rationales.
            rationale_num -- suggested number of rationales.
        Outputs:
            continuity_loss --  \sum_{i} | z_{i-1} - z_{i} |.
            sparsity_loss -- |mean(z_{i}) - percent|.
        """

        if mask is not None:
            mask_z = z * mask  # (batch_size,).
            seq_lengths = torch.sum(mask, dim=1)
        else:
            mask_z = z
            seq_lengths = torch.sum(z - z + 1.0, dim=1)

        mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)

        continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths  # (batch_size,)
        percentage = rationale_num * 2 / seq_lengths # two transitions from rationale to not.
        continuity_loss = torch.abs(continuity_ratio - percentage)

        sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths  # (batch_size,).
        percentage = rationale_len / seq_lengths
        sparsity_loss = torch.abs(sparsity_ratio - percentage)

        return continuity_loss, sparsity_loss


    def _get_classifier_loss(self, predict, y):
        """
        Get loss and accuracy for classifier or anti-classifier.
        Inputs:
        Outputs:
        """
        loss_c = torch.mean(self.loss_func(predict, y))
        accuracy_c = (torch.max(predict, dim=1)[1] == y).float()
        if self.use_cuda:
            accuracy_c = accuracy_c.cuda()
        return loss_c, accuracy_c


    def train_one_step(self, x, y, m):
        """
        Train one step of the model from x, y and m; and backpropagate errors.
        Inputs:
            x -- Variable() of input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            y -- Variable() of output y, shape (batch_size,),
                 each element in the batch is an integer representing the label.
            m -- Variable() of mask m, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            loss_val -- list of losses, [classifier, anti_classifier, tagger].
            predict -- prediction score of classifier, shape (batch_size, |label|),
                       each element at i is a predicted probability for label[i].
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        """

        # Forward model and get rationales, predictions, etc.
        predict, anti_predict, z, neg_log_probs = self.forward(x, m)

        # Get loss and accuracy for classifier and anti-classifier.
        loss_classifier, accuracy_classifier = self._get_classifier_loss(predict, y)
        loss_anti_classifier, accuracy_anti_classifier = self._get_classifier_loss(anti_predict, y)

        # Get regularization loss for tagged rationales.
        loss_continuity, loss_sparsity = self._get_regularization_loss(z, self.rationale_len, self.rationale_num, m)

        # Get reinforce-style loss for tagger.
        loss_tagger = self._get_tagger_loss(accuracy_classifier, accuracy_anti_classifier,
                                            loss_continuity, loss_sparsity,
                                            z, neg_log_probs, m)

        # Backpropagate losses.
        losses = [loss_classifier, loss_anti_classifier, loss_tagger]
        opts = [self.opt_classifier, self.opt_anti_classifier, self.opt_tagger]
        loss_val = []
        for loss, opt in zip(losses, opts):
            loss.backward()
            opt.step()
            opt.zero_grad()
            loss_val.append(loss.item())
        
        return loss_val, predict, z


# Test for Rationalizer.
def test_rationalizer(args):

    embeddings = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]).astype(np.float32)
    args.num_labels = 2
    args.embedding_dim = 4
    args.hidden_dim = 6
    args.head_num = 2

    model = Rationalizer(embeddings, args)
    if args.cuda:
        model.cuda()
    
    model.train()
    x = Variable(torch.tensor([[1, 3, 3, 2, 2], [2, 1, 3, 1, 0], [3, 1, 2, 0, 0]]))  # (batch_size, seq_len).
    y = Variable(torch.tensor([1, 0, 1]))  # (batch_size,).
    m = Variable(torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]))  # (batch_size, seq_len).
    if args.cuda:
        x = x.cuda()
        y = y.cuda()
        m = m.cuda()

    loss_val, predict, z = model.train_one_step(x, y, m)
    print(loss_val, predict, z)
