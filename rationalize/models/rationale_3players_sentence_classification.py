# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

from modules.nn import RnnModel
from modules.generator import Generator

from models.models_utils import bao_regularization_loss_batch

from collections import deque


def count_regularization_baos_for_both(z, count_tokens, count_pieces, mask=None):
    """
    Compute regularization loss, based on a given rationale sequence
    Use Yujia's formulation

    Inputs:
        z -- torch variable, "binary" rationale, (batch_size, sequence_length)
        percentage -- the percentage of words to keep
    Outputs:
        a loss value that contains two parts:
        continuity_loss --  \sum_{i} | z_{i-1} - z_{i} | 
        sparsity_loss -- |mean(z_{i}) - percent|
    """

    # (batch_size,)
    if mask is not None:
        mask_z = z * mask
        seq_lengths = torch.sum(mask, dim=1)
    else:
        mask_z = z
        seq_lengths = torch.sum(z - z + 1.0, dim=1)
    
    mask_z_ = torch.cat([mask_z[:, 1:], mask_z[:, -1:]], dim=-1)
        
    continuity_ratio = torch.sum(torch.abs(mask_z - mask_z_), dim=-1) / seq_lengths #(batch_size,) 
    percentage = count_pieces * 2 / seq_lengths
#     continuity_loss = F.threshold(continuity_ratio - percentage, 0, 0, False)
    continuity_loss = torch.abs(continuity_ratio - percentage)
    
    sparsity_ratio = torch.sum(mask_z, dim=-1) / seq_lengths #(batch_size,)
    percentage = count_tokens / seq_lengths #(batch_size,)
#     sparsity_loss = F.threshold(sparsity_ratio - percentage, 0, 0, False)
    sparsity_loss = torch.abs(sparsity_ratio - percentage)

    return continuity_loss, sparsity_loss


class ClassifierModule(nn.Module):
    '''
    classifier for both E and E_anti models
    '''
    def __init__(self, args):
        super(ClassifierModule, self).__init__()
        self.args = args
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        self.input_dim = args.embedding_dim
        
        self.encoder = RnnModel(self.args, self.input_dim)
        self.predictor = nn.Linear(self.hidden_dim, self.num_labels)
        
        self.NEG_INF = -1.0e6
        

    def forward(self, word_embeddings, z, mask):
        """
        Inputs:
            word_embeddings -- torch Variable in shape of (batch_size, length, embed_dim)
            z -- rationale (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
        """        

        masked_input = word_embeddings * z.unsqueeze(-1)
        hiddens = self.encoder(masked_input, mask)
        
        max_hidden = torch.max(hiddens + (1 - mask * z).unsqueeze(1) * self.NEG_INF, dim=2)[0]
        
        predict = self.predictor(max_hidden)

        return predict


class Rationale3PlayerClassificationModel(nn.Module):
    
    def __init__(self, embeddings, args):
        super(Rationale3PlayerClassificationModel, self).__init__()
        self.args = args
        
        self.model_type = args.model_type
        self.use_cuda = args.cuda
        self.lambda_sparsity = args.lambda_sparsity
        self.lambda_continuity = args.lambda_continuity
        self.lambda_anti = args.lambda_anti
        
        self.NEG_INF = -1.0e6
                    
        self.vocab_size, self.embedding_dim = embeddings.shape
        self.embed_layer = self._create_embed_layer(embeddings)
        
        self.num_labels = args.num_labels
        self.hidden_dim = args.hidden_dim
        self.mlp_hidden_dim = args.mlp_hidden_dim #50
        
        self.input_dim = args.embedding_dim
        
        self.E_model = ClassifierModule(args)
        self.E_anti_model = ClassifierModule(args)
        
        self.loss_func = nn.CrossEntropyLoss()
        
        
    def _create_embed_layer(self, embeddings):
        embed_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        embed_layer.weight.data = torch.from_numpy(embeddings)
        embed_layer.weight.requires_grad = self.args.fine_tuning
        return embed_layer
        
    def forward(self, x, e, mask):
        pass


class HardRationale3PlayerClassificationModel(Rationale3PlayerClassificationModel):
    
    def __init__(self, embeddings, args):
        super(HardRationale3PlayerClassificationModel, self).__init__(embeddings, args)
        self.generator = Generator(args, self.input_dim)
        self.highlight_percentage = args.highlight_percentage
        self.highlight_count = args.highlight_count
        self.exploration_rate = args.exploration_rate
        
        self.loss_func = nn.CrossEntropyLoss(reduce=False)
        
        if args.margin is not None:
            self.margin = args.margin

        
    def init_optimizers(self):
        self.opt_E = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_model.parameters()), lr=self.args.lr)
        self.opt_E_anti = torch.optim.Adam(filter(lambda x: x.requires_grad, self.E_anti_model.parameters()), lr=self.args.lr)
        
    def init_rl_optimizers(self):
        self.opt_G_rl = torch.optim.Adam(filter(lambda x: x.requires_grad, self.generator.parameters()), lr=self.args.lr * 0.1)
        
        
    def init_reward_queue(self):
        queue_length = 200
        self.z_history_rewards = deque(maxlen=queue_length)
        self.z_history_rewards.append(0.)
        
    def init_C_model(self):
        self.C_model = ClassifierModule(self.args)
        
    def get_C_model_pred(self, x, mask):
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        z_ones = torch.ones_like(x).type(torch.cuda.FloatTensor)
        cls_predict = self.C_model(word_embeddings, z_ones, mask)
        return cls_predict
        
    def _generate_rationales(self, z_prob_):
        '''
        Input:
            z_prob_ -- (num_rows, length, 2)
        Output:
            z -- (num_rows, length)
        '''        
        z_prob__ = z_prob_.view(-1, 2) # (num_rows * length, 2)
        
        # sample actions
        sampler = torch.distributions.Categorical(z_prob__)
        if self.training:
            z_ = sampler.sample() # (num_rows * p_length,)
        else:
            z_ = torch.max(z_prob__, dim=-1)[1]
        
        #(num_rows, length)
        z = z_.view(z_prob_.size(0), z_prob_.size(1))
        
        if self.use_cuda == True:
            z = z.type(torch.cuda.FloatTensor)
        else:
            z = z.type(torch.FloatTensor)
            
        # (num_rows * length,)
        neg_log_probs_ = -sampler.log_prob(z_)
        # (num_rows, length)
        neg_log_probs = neg_log_probs_.view(z_prob_.size(0), z_prob_.size(1))
        
        return z, neg_log_probs
    
    
    def train_gen_one_step(self, x, label, mask):
        z_baseline = Variable(torch.FloatTensor([float(np.mean(self.z_history_rewards))]))
        if self.args.cuda:
            z_baseline = z_baseline.cuda()
        
        self.opt_G_rl.zero_grad()
        
        predict, anti_predict, z, neg_log_probs = self.forward(x, mask)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, z, 
                                                                         neg_log_probs, z_baseline, 
                                                                         mask, label)
        
        losses = {'g_rl_loss':rl_loss.cpu().data}

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
    
        z_batch_reward = np.mean(rewards.cpu().data.numpy())
        self.z_history_rewards.append(z_batch_reward)
        
        return losses, predict, anti_predict, z, rewards, continuity_loss, sparsity_loss

    
    def train_one_step(self, x, label, baseline, mask):
        
        predict, anti_predict, z, neg_log_probs = self.forward(x, mask)
        
        e_loss_anti = torch.mean(self.loss_func(anti_predict, label))
        
        e_loss = torch.mean(self.loss_func(predict, label))
        
        rl_loss, rewards, continuity_loss, sparsity_loss = self.get_loss(predict, anti_predict, z, 
                                                                         neg_log_probs, baseline, 
                                                                         mask, label)
        
        losses = {'e_loss':e_loss.cpu().data, 'e_loss_anti':e_loss_anti.cpu().data,
                 'g_loss':rl_loss.cpu().data}
        
        e_loss_anti.backward()
        self.opt_E_anti.step()
        self.opt_E_anti.zero_grad()
        
        e_loss.backward()
        self.opt_E.step()
        self.opt_E.zero_grad()

        rl_loss.backward()
        self.opt_G_rl.step()
        self.opt_G_rl.zero_grad()
        
        return losses, predict, anti_predict, z, rewards, continuity_loss, sparsity_loss
    
        
    def forward(self, x, mask):
        """
        Inputs:
            x -- torch Variable in shape of (batch_size, length)
            mask -- torch Variable in shape of (batch_size, length)
        Outputs:
            predict -- (batch_size, num_label)
            z -- rationale (batch_size, length)
        """        
        word_embeddings = self.embed_layer(x) #(batch_size, length, embedding_dim)
        
        neg_inf = -1.0e6
        
        z_scores_ = self.generator(word_embeddings, mask) #(batch_size, length, 2)
        z_scores_[:, :, 1] = z_scores_[:, :, 1] + (1 - mask) * neg_inf

        z_probs_ = F.softmax(z_scores_, dim=-1)
        
        z_probs_ = (mask.unsqueeze(-1) * ( (1 - self.exploration_rate) * z_probs_ + self.exploration_rate / z_probs_.size(-1) ) ) + ((1 - mask.unsqueeze(-1)) * z_probs_)
        
        z, neg_log_probs = self._generate_rationales(z_probs_)
        
        predict = self.E_model(word_embeddings, z, mask)
        
        anti_predict = self.E_anti_model(word_embeddings, 1 - z, mask)

        return predict, anti_predict, z, neg_log_probs

    
    def get_advantages(self, predict, anti_predict, label, z, neg_log_probs, baseline, mask):
        '''
        Input:
            z -- (batch_size, length)
        '''
        
        # total loss of accuracy (not batchwise)
        _, y_pred = torch.max(predict, dim=1)
        prediction = (y_pred == label).type(torch.FloatTensor)
        _, y_anti_pred = torch.max(anti_predict, dim=1)
        prediction_anti = (y_anti_pred == label).type(torch.FloatTensor) * self.lambda_anti
        if self.use_cuda:
            prediction = prediction.cuda()  #(batch_size,)
            prediction_anti = prediction_anti.cuda()
        
        continuity_loss, sparsity_loss = bao_regularization_loss_batch(z, self.highlight_percentage, mask)
#         continuity_loss, sparsity_loss = bao_regularization_hinge_loss_batch(z, self.highlight_percentage, mask)
#         continuity_loss, sparsity_loss = count_regularization_hinge_loss_batch(z, self.highlight_count, mask)
#         continuity_loss, sparsity_loss = bao_regularization_hinge_loss_batch_with_none_loss(z, self.highlight_percentage, 
#                                                                              self.none_relation_id, mask)
        
        continuity_loss = continuity_loss * self.lambda_continuity
        sparsity_loss = sparsity_loss * self.lambda_sparsity

        # batch RL reward
        rewards = prediction - prediction_anti - sparsity_loss - continuity_loss
        
        advantages = rewards - baseline # (batch_size,)
        advantages = Variable(advantages.data, requires_grad=False)
        if self.use_cuda:
            advantages = advantages.cuda()
        
        return advantages, rewards, continuity_loss, sparsity_loss
    
    def get_loss(self, predict, anti_predict, z, neg_log_probs, baseline, mask, label):
        reward_tuple = self.get_advantages(predict, anti_predict, label, z, neg_log_probs, baseline, mask)
        advantages, rewards, continuity_loss, sparsity_loss = reward_tuple
        
        # (batch_size, q_length)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)
        rl_loss = torch.sum(neg_log_probs * advantages_expand_ * mask)
        
        return rl_loss, rewards, continuity_loss, sparsity_loss

