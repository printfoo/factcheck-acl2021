# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from collections import deque

from models.nn import RnnModel
from models.tagger import Tagger
from models.classifier import Classifier


class RationaleClassification(nn.Module):
    """
    Rationale classification model.
    Using model.Classifier and model.Generator modules.
    """

    def __init__(self, embeddings, args):
        super(RationaleClassification, self).__init__()
        self.NEG_INF = -1.0e6
        self.args = args

        self.use_cuda = args.cuda
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti

        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings)

        self.classifier = Classifier(args)
        self.anti_classifier = Classifier(args)
        self.tagger = Tagger(args, self.embedding_dim)

        self.opt_c = torch.optim.Adam(filter(lambda x: x.requires_grad, self.classifier.parameters()), lr=args.lr)
        self.opt_anti_c = torch.optim.Adam(filter(lambda x: x.requires_grad, self.anti_classifier.parameters()), lr=args.lr)
        self.opt_t_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.tagger.parameters()), lr=args.lr*0.1)

        self.exploration_rate = args.exploration_rate
        self.rationale_len = args.rationale_len
        self.rationale_num = args.rationale_num

        self.z_history_rewards = deque(maxlen=200)
        self.z_history_rewards.append(0.)

        self.loss_func = nn.CrossEntropyLoss(reduce=False)


    def _create_embed_layer(self, embeddings):
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = bool(self.args.fine_tuning)
        return embed_layer


    def _generate_rationales(self, z_prob_):
        """
        Input:
            z_prob_ -- (num_rows, length, 2).
        Output:
            z -- (num_rows, length).
        """
        z_prob__ = z_prob_.view(-1, 2)  # (num_rows*length, 2).

        sampler = torch.distributions.Categorical(z_prob__)  # Sample actions.
        if self.training:
            z_ = sampler.sample()  # (num_rows*p_length,).
        else:
            z_ = torch.max(z_prob__, dim=-1)[1]

        z = z_.view(z_prob_.size(0), z_prob_.size(1))  # (num_rows, length).
        
        if self.use_cuda == True:
            z = z.type(torch.cuda.FloatTensor)
        else:
            z = z.type(torch.FloatTensor)

        neg_log_probs_ = -sampler.log_prob(z_)  # (num_rows*length,).
        neg_log_probs = neg_log_probs_.view(z_prob_.size(0), z_prob_.size(1))  # (num_rows, length).
        
        return z, neg_log_probs


    def forward(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length).
            mask -- torch Variable in shape of (batch_size, length).
        Outputs:
            predict -- (batch_size, num_label).
            z -- rationale (batch_size, length).
        """
        word_embeddings = self.embed_layer(x)  # (batch_size, length, embedding_dim).

        z_scores_ = self.tagger(word_embeddings, mask)  # (batch_size, length, 2).
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * self.NEG_INF

        z_probs_ = F.softmax(z_scores_, dim=-1)
        z_probs_ = (mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - mask.unsqueeze(-1)) * z_probs_)

        z, neg_log_probs = self._generate_rationales(z_probs_)

        predict = self.classifier(word_embeddings, z, mask)
        anti_predict = self.anti_classifier(word_embeddings, 1 - z, mask)

        return predict, anti_predict, z, neg_log_probs
    

    def _regularization_loss_batch(self, z, rationale_len, rationale_num, mask=None):
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


    def _get_advantages(self, predict, anti_predict, label, z, neg_log_probs, baseline, mask):
        """
        Input:
            z -- (batch_size, length).
        """
        
        # Total loss of accuracy (not batchwise).
        _, y_pred = torch.max(predict, dim=1)
        prediction = (y_pred == label).type(torch.FloatTensor)
        _, y_anti_pred = torch.max(anti_predict, dim=1)
        prediction_anti = (y_anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        if self.use_cuda:
            prediction = prediction.cuda()  # (batch_size,).
            prediction_anti = prediction_anti.cuda()
        
        continuity_loss, sparsity_loss = self._regularization_loss_batch(z, self.rationale_len, self.rationale_num, mask)
        
        continuity_loss *= self.lambda_continuity
        sparsity_loss *= self.lambda_sparsity

        rewards = prediction - prediction_anti - sparsity_loss - continuity_loss  # batch RL reward.

        advantages = rewards - baseline  # (batch_size,).
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()
        
        return advantages, rewards, continuity_loss, sparsity_loss


    def _get_loss(self, predict, anti_predict, z, neg_log_probs, baseline, mask, label):
        reward_tuple = self._get_advantages(predict, anti_predict, label, z, neg_log_probs, baseline, mask)
        advantages, rewards, continuity_loss, sparsity_loss = reward_tuple

        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)  # (batch_size, q_length).
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)
        
        return rl_loss, rewards, continuity_loss, sparsity_loss


    def train_one_step(self, x, label, mask):

        baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
        if self.use_cuda:
            baseline = baseline.cuda()

        predict, anti_predict, z, neg_log_probs = self.forward(x, mask)
        
        loss_anti_c = torch.mean(self.loss_func(anti_predict, label))        
        loss_c = torch.mean(self.loss_func(predict, label))
        loss_tuple = self._get_loss(predict, anti_predict, z, neg_log_probs, baseline, mask, label)
        rl_loss, rewards, continuity_loss, sparsity_loss = loss_tuple
        
        z_batch_reward = torch.mean(rewards).item()
        self.z_history_rewards.append(z_batch_reward)

        # Backprop loss to predictor E.
        loss_anti_c.backward()
        self.opt_anti_c.step()
        self.opt_anti_c.zero_grad()
        
        # Backprop loss to predictor anti_E.
        loss_c.backward()
        self.opt_c.step()
        self.opt_c.zero_grad()

        # Backprop loss to generator G.
        rl_loss.backward()
        self.opt_t_rl.step()
        self.opt_t_rl.zero_grad()
        
        losses = {"loss_c": loss_c.data, "loss_anti_c": loss_anti_c.data, "loss_t": rl_loss.data}
        return losses, predict, z
    
