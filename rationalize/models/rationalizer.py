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

        # General parameters.
        self.NEG_INF = -1.0e6
        self.use_cuda = args.cuda
        p_grad = lambda module: filter(lambda _: _.requires_grad, module.parameters())
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

        # Initialize embedding layers.
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings, bool(args.fine_tuning))

        # Initialize classifiers.
        self.classifier = Classifier(args)
        self.opt_classifier = torch.optim.Adam(p_grad(self.classifier), lr=args.lr)

        # Whether to tag rationale, otherwise just standard classification problem.
        self.rationale_tagger = bool(args.rationale_tagger)
        if self.rationale_tagger:
            self.tagger = Tagger(args)
            self.opt_tagger = torch.optim.Adam(p_grad(self.tagger), lr=args.lr*0.1)
            self.history_rewards = deque(maxlen=200) # Initialize reward queue for reinforce loss.
            self.history_rewards.append(0.)

        # Whether and how much to use an anti predictor to limit rationale selection.
        if bool(args.anti_predictor):
            self.anti_classifier = Classifier(args)
            self.lambda_anti = args.lambda_anti
            self.opt_anti_classifier = torch.optim.Adam(p_grad(self.anti_classifier), lr=args.lr)
        else:
            self.lambda_anti = 0

        # Whether and how much to use linear signal to guide rationale selection.
        if bool(args.linear_signal):
            self.lambda_s = args.lambda_s
            self.threshold_s = args.threshold_s
        else:
            self.lambda_s = 0
            self.threshold_s = 0

        # Whether and how much to use domain knowledge to guide rationale selection.
        if bool(args.domain_knowledge):
            self.lambda_d = args.lambda_d
        else:
            self.lambda_d = 0

        # Whether and how much to use regulation on rationale selection.
        if bool(args.rationale_regulation):
            self.lambda_sparsity = args.lambda_sparsity
            self.lambda_continuity = args.lambda_continuity
            self.rationale_len = args.rationale_len
            self.rationale_num = args.rationale_num
        else:
            self.lambda_sparsity = 0
            self.lambda_continuity = 0
            self.rationale_len = 0
            self.rationale_num = 0


    def _create_embed_layer(self, embeddings, fine_tuning=False):
        """
        Create a lookup layer for embeddings.
        Input:
            embeddings -- embeddings of tokens, shape (|vocab|, embedding_dim). 
        Output:
            embed_layer -- a lookup layer for embeddings,
                           inputs word index and returns word embedding.
        """
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = bool(fine_tuning)
        return embed_layer


    def forward(self, x, m):
        """
        Forward model from x and m and get rationales, predictions, etc.
        Inputs:
            x -- input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            m -- mask m, shape (batch_size, seq_len),
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
        if self.rationale_tagger:
            z, neg_log_probs = self.tagger(embeddings, m)
        else:
            z, neg_log_probs = m, None

        # Prediction of anti classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        if self.lambda_anti:
            anti_predict = self.anti_classifier(embeddings, 1 - z, m)
        else:
            anti_predict = None

        # Prediction of classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        predict = self.classifier(embeddings, z, m)

        return predict, anti_predict, z, neg_log_probs
    

    def _get_tagger_loss(self, reward_classifier, reward_anti_classifier, reward_s, reward_d,
                         loss_continuity, loss_sparsity, z, neg_log_probs, m):
        """
        Get reinforce-style loss for tagger.
        Inputs:
            reward_classifier -- accuracy of the classifier, shape (batch_size,),
                                   each element at i is of 0/1 for incorrect/correct prediction.
            reward_anti_classifier -- accuracy of the anti_classifier, shape (batch_size,),
            reward_s -- reward of z comparing to linear signal s, shape (batch_size,),
            reward_d -- reward of z comparing to domain knowledge d, shape (batch_size,),
            loss_continuity -- loss for continuity, shape (batch_size,),
            sparsity_loss -- loss for sparsity, shape (batch_size,),
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            neg_log_probs -- negative log probability, shape (batch_size, seq_len).
            m -- mask m, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            loss_tagger -- reinforce-style loss for tagger, shape (batch_size, seq_len).
        """

        # Mean of history rewards as baseline,
        # (|history|,) -> (1,).
        history_rewards_mean = Variable(torch.tensor(np.mean(self.history_rewards)))
        if self.use_cuda:
            history_rewards_mean = history_rewards_mean.cuda()

        # Reinforce-style loss for this batch,
        # (batch_size,) -> (batch_size,).
        rewards = (reward_classifier
                   - reward_anti_classifier * self.lambda_anti
                   - loss_continuity * self.lambda_continuity 
                   - loss_sparsity * self.lambda_sparsity
                   + reward_s * self.lambda_s
                   + reward_d * self.lambda_d)

        # Update mean loss of this batch to the history reward queue.
        self.history_rewards.append(torch.mean(rewards).item())

        # Get advantages of this run over history.
        # (batch_size,) -> (batch_size,).
        advantages = rewards - history_rewards_mean
        advantages.requires_grad = False
        if self.use_cuda:
            advantages = advantages.cuda()

        # Expand advantages to the same shape of z, by copying its value to seq_len,
        # (batch_size,) -> (batch_size, seq_len).
        advantages = advantages.unsqueeze(-1).expand_as(neg_log_probs)

        # Sum up advantages by the sequence with neg_log_probs, and by the batch,
        # (batch_size, seq_len) -> (1,).
        loss_tagger = torch.sum(neg_log_probs * advantages * m)

        return loss_tagger


    def _get_guidance_reward(self, z, ref):
        """
        Get guidance reward of rationale selection.
        Inputs:
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            ref -- reference for rationale selection, shape (batch_size, seq_len),
                   each element is of 0/1 selecting a token or not,
                   this reference could be linear signal s or domain knowledge d.
        Outputs:
            reward -- reward of z comparing to ref, shape (batch_size,),
        """

        # Get reward of rationale selection comparing to reference,
        # (batch_size, seq_len) -> (batch_size,)
        reward = torch.mean(z * ref, dim=1)
        return reward


    def _get_regularization_loss(self, z, m=None):
        """
        Get regularization loss of rationale selection.
        Inputs:
            z -- selected rationale, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            m -- mask m, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
        Outputs:
            loss_continuity -- loss for continuity, shape (batch_size,),
            loss_sparsity -- loss for sparsity, shape (batch_size,),
        """

        # Get sequence lengths and masked rationales.
        if m is not None:
            mask_z = z * m
            seq_lens = torch.sum(m, dim=1)
        else:
            mask_z = z
            seq_lens = torch.sum(z - z + 1.0, dim=1)

        # Shift masked z by one to the left: z[i-1] = z[i],
        # Then, get the number of transitions (2 * the number of rationales), and normalize by seq_len,
        # Then, get loss for continuity: the difference of rationale ratio this run v.s. recommended,
        # (batch_size, seq_len) -> (batch_size,).
        mask_z_shift_left = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        ratio_continuity = torch.sum(torch.abs(mask_z - mask_z_shift_left), dim=-1) / seq_lens
        ratio_recommend = self.rationale_num * 2 / seq_lens
        loss_continuity = torch.abs(ratio_continuity - ratio_recommend)

        # Get the length of all selected rationales, and normalize by seq_len,
        # Then, get loss for sparsity: the difference of rationale len this run v.s. recommended,
        # (batch_size, seq_len) -> (batch_size,).
        ratio_sparsity = torch.sum(mask_z, dim=-1) / seq_lens
        ratio_recommend = self.rationale_len / seq_lens
        loss_sparsity = torch.abs(ratio_sparsity - ratio_recommend)

        return loss_continuity, loss_sparsity


    def _get_classifier_loss(self, predict, y):
        """
        Get loss and accuracy for classifier or anti-classifier.
        Inputs:
            predict -- prediction score of classifier, shape (batch_size, |label|),
                       each element at i is a predicted probability for label[i].
            y -- output y, shape (batch_size,),
                 each element in the batch is an integer representing the label.
        Outputs:
            loss_classifier -- loss of the classifier, shape (1,),
                               a scala averages the loss of this batch.
            reward_classifier -- reward of the classifier, shape (batch_size,),
                                 each element at i is of 0/1 for incorrect/correct prediction.
        """

        # Get loss of the classifier for the entire batch,
        # (batch_size,) -> (1,)
        loss_classifier = torch.mean(self.loss_func(predict, y))
        
        # Get accuracy of the classifier for each input, 
        # (batch_size,) -> (batch_size,)
        reward_classifier = (torch.max(predict, dim=1)[1] == y).float()
        
        return loss_classifier, reward_classifier


    def train_one_step(self, x, y, m, r, s, d):
        """
        Train one step of the model from x, y and m; and backpropagate errors.
        Inputs:
            x -- input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            y -- output y, shape (batch_size,),
                 each element in the batch is an integer representing the label.
            m -- mask m, shape (batch_size, seq_len),
                 each element in the seq_len is of 0/1 selecting a token or not.
            r -- rationale annotation r, shape (batch_size, seq_len),
                 each element is of 0/1 if a word is selected as rationale by human annotators.
            s -- linear signal s, shape (batch_size, seq_len),
                 each element is from -1-1 of coefficient of linear regression.
            d -- domain knowledge d, shape (batch_size, seq_len),
                 each element is of -1/0/1 if a word is neg/non/pos-rationale with domain knowledge.
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
        loss_classifier, reward_classifier = self._get_classifier_loss(predict, y)
        if self.lambda_anti:
            loss_anti_classifier, reward_anti_classifier = self._get_classifier_loss(anti_predict, y)
        else:
            loss_anti_classifier, reward_anti_classifier = 0, 0
        
        # If use rationale tagger.
        if self.rationale_tagger:

            # Get regularization loss for tagged rationales.
            loss_continuity, loss_sparsity = self._get_regularization_loss(z, m)
        
            # Get linear signal reward for tagged rationales.
            if self.lambda_s:
                s = (s >= self.threshold_s).float()  # Binary.
                reward_s = self._get_guidance_reward(z, s)
            else:
                reward_s = 0

            # Get domain knowledge reward for tagged rationales.
            if self.lambda_d:
                d = torch.abs(d)  # Binary.
                reward_d = self._get_guidance_reward(z, d)
            else:
                reward_d = 0

            # Get reinforce-style loss for tagger.
            loss_tagger = self._get_tagger_loss(reward_classifier, reward_anti_classifier, reward_s, reward_d,
                                                loss_continuity, loss_sparsity, z, neg_log_probs, m)

        # Backpropagate losses.
        losses = [loss_classifier]
        opts = [self.opt_classifier]
        if self.rationale_tagger:  # Append tagger loss and optimizer.
            losses.append(loss_tagger)
            opts.append(self.opt_tagger)
        if self.lambda_anti:  # Append anti classifier loss and optimizer.
            losses.append(loss_anti_classifier)
            opts.append(self.opt_anti_classifier)
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
    r = Variable(torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]))  # (batch_size, seq_len).
    s = Variable(torch.tensor([[0.1, 1, -0.1, 0, 0], [1, -0.1, 0, 0, 0], [0, 0, 0.1, 0, 0]]))  # (batch_size, seq_len).
    d = Variable(torch.tensor([[0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]]))  # (batch_size, seq_len).
    if args.cuda:
        x = x.cuda()
        y = y.cuda()
        m = m.cuda()
        x = x.cuda()
        r = r.cuda()
        s = s.cuda()
        d = d.cuda()

    loss_val, predict, z = model.train_one_step(x, y, m, r, s, d)
    print(loss_val, predict, z)
