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

        # Initialize parameters.
        self.NEG_INF = -1.0e6
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
        self.tagger = Tagger(args, self.embedding_dim)
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

        # Initialize optimizers.
        p_grad = lambda module: filter(lambda _: _.requires_grad, module.parameters())
        self.opt_c = torch.optim.Adam(p_grad(self.classifier), lr=args.lr)
        self.opt_anti_c = torch.optim.Adam(p_grad(self.anti_classifier), lr=args.lr)
        self.opt_t_rl = torch.optim.Adam(p_grad(self.tagger), lr=args.lr*0.1)

        # Initialize reward queue for reinforce loss.
        self.z_history_rewards = deque(maxlen=200)
        self.z_history_rewards.append(0.)


    def _create_embed_layer(self, embeddings, fine_tuning=False):
        """
        Create a lookup layer for embeddings.
        Input:
            embeddings -- embeddings of tokens, shape (|vocab|, embedding_dim). 
        Output:
            embed_layer -- a lookup layer for embeddings.
                           inputs token' ID and returns token's embedding.
        """
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = fine_tuning
        return embed_layer


    def _generate_rationales(self, z_probs):
        """
        Input:
            z_prob_ -- probability of selecting rationale, shape (batch_size, seq_len, 2),
                       each element is a 0-1 probability of selecting a token or not.
        Output:
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        """

        # Reshape z_probs by concatenating all batches,
        # (batch_size, seq_len, 2) -> (batch_size * seq_len, 2)
        z_probs_all = z_probs.view(-1, 2)

        # Create a categorical distribution parameterized by concatenated probs.
        sampler = torch.distributions.Categorical(z_probs_all)

        if self.training:  # If train, sample rationale from the distribution.
            z_all = sampler.sample()  # (batch_size * seq_len).
        else:  # If eval, use max prob as rationale.
            z_all = torch.max(z_probs_all, dim=-1)[1]  # (batch_size * seq_len).
        neg_log_probs_all = -sampler.log_prob(z_all)  # (batch_size * seq_len).

        # Recover concatenated rationales to each batch,
        # (batch_size * seq_len) -> (batch_size, seq_len).
        z = z_all.view(z_probs.size(0), z_probs.size(1))
        neg_log_probs = neg_log_probs_all.view(z_probs.size(0), z_probs.size(1))

        return z.float(), neg_log_probs


    def forward(self, x, m):
        """
        Inputs:
            x -- Variable() of input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            m -- Variable() of mask m, shape (batch_size, seq_len).
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
        word_embeddings = self.embed_layer(x)

        # Pass embeddings through the tagger module,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, 2).
        z_scores = self.tagger(word_embeddings, m)
        
        # Replace (batch_size, seq_len, 1) with -inf.
        z_scores[:, :, 1] = z_scores[:, :, 1] + (1 - m) * self.NEG_INF

        # Run a softmax for valid probs (batch_size, seq_len, 0) + (batch_size, seq_len, 1) = 1.
        z_probs = F.softmax(z_scores, dim=-1)

        # Generate rationale and negative log probs,
        # (batch_size, seq_len, 2) -> (batch_size, seq_len)
        z, neg_log_probs = self._generate_rationales(z_probs)

        # Prediction of classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        predict = self.classifier(word_embeddings, z, m)
        
        # Prediction of anti classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        anti_predict = self.anti_classifier(word_embeddings, 1 - z, m)

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


# Test for RationaleClassification.
def test_rationale_classification(args):

    embeddings = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]).astype(np.float32)
    args.num_labels = 2
    args.embedding_dim = 3

    model = RationaleClassification(embeddings, args)
    if args.cuda:
        model.cuda()
    
    model.train()
    x = Variable(torch.tensor([[1, 3, 3, 2], [2, 1, 3, 0], [3, 1, 2, 0]]))  # (batch_size, seq_len).
    y = Variable(torch.tensor([1, 0, 1]))  # (batch_size,).
    m = Variable(torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0]]))  # (batch_size, seq_len).
    if args.cuda:
        x = x.cuda()
        y = y.cuda()
        m = m.cuda()

    losses, predict, z = model.train_one_step(x, y, m)
    print(losses, predict, z)
