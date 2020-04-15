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
        self.history_rewards = deque(maxlen=200)
        self.history_rewards.append(0.)


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
        embed_layer.weight.requires_grad = fine_tuning
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
        z, neg_log_probs = self.tagger(embeddings, m)

        # Prediction of classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        predict = self.classifier(embeddings, z, m)
        
        # Prediction of anti classifier,
        # (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, |label|)
        anti_predict = self.anti_classifier(embeddings, 1 - z, m)

        return predict, anti_predict, z, neg_log_probs
    

    def _get_tagger_loss(self, accuracy_classifier, accuracy_anti_classifier,
                         loss_continuity, loss_sparsity, z, neg_log_probs, m):
        """
        Get reinforce-style loss for tagger.
        Inputs:
            accuracy_classifier -- accuracy of the classifier, shape (batch_size,),
                                   each element at i is of 0/1 for incorrect/correct prediction.
            accuracy_anti_classifier -- accuracy of the anti_classifier, shape (batch_size,),
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
        rewards = (accuracy_classifier
                   - accuracy_anti_classifier * self.lambda_anti
                   - loss_continuity * self.lambda_continuity 
                   - loss_sparsity * self.lambda_sparsity)

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
            accuracy_classifier -- accuracy of the classifier, shape (batch_size,),
                                   each element at i is of 0/1 for incorrect/correct prediction.
        """

        # Get loss of the classifier for the entire batch,
        # (batch_size,) -> (1,)
        loss_classifier = torch.mean(self.loss_func(predict, y))
        
        # Get accuracy of the classifier for each input, 
        # (batch_size,) -> (batch_size,)
        accuracy_classifier = (torch.max(predict, dim=1)[1] == y).float()
        if self.use_cuda:
            accuracy_classifier = accuracy_classifier.cuda()
        
        return loss_classifier, accuracy_classifier


    def train_one_step(self, x, y, m):
        """
        Train one step of the model from x, y and m; and backpropagate errors.
        Inputs:
            x -- input x, shape (batch_size, seq_len),
                 each element in the seq_len is of 0-|vocab| pointing to a token.
            y -- output y, shape (batch_size,),
                 each element in the batch is an integer representing the label.
            m -- mask m, shape (batch_size, seq_len),
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
        loss_continuity, loss_sparsity = self._get_regularization_loss(z, m)

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
