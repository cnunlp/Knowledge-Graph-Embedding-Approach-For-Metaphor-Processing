from Model import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransH(Model):
    def __init__(self, config):
        super(TransH, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.entityTotal+1, self.ent_size, padding_idx = 0)
        self.rel_embeddings = nn.Embedding(self.propertyTotal+1, self.rel_size, padding_idx=0)
        self.norm_vector = nn.Embedding(self.propertyTotal+1, self.rel_size, padding_idx=0)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        # random init
        self.ent_embeddings.weight = nn.Parameter(torch.cat((torch.zeros(1, self.ent_size), torch.randn(self.entityTotal, self.ent_size)), 0))
        self.rel_embeddings.weight = nn.Parameter(torch.cat((torch.zeros(1, self.rel_size), torch.randn(self.propertyTotal, self.rel_size)), 0))
        self.norm_vector.weight = nn.Parameter(torch.cat((torch.zeros(1, self.rel_size), torch.randn(self.propertyTotal, self.rel_size)), 0))

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data[1:])
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data[1:])
        nn.init.xavier_uniform_(self.norm_vector.weight.data[1:])

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return e - torch.sum(e * norm, -1, True) * norm

    def _calc(self, h, t, r):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        score = torch.norm((h + r - t), self.p_norm, dim=-1)  # batch
        return score

    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.config.margin, False)
        y = torch.Tensor([-1])
        if self.config.use_cuda:
            criterion = criterion.cuda()
            y = y.cuda()
        loss = criterion(p_score, n_score, y)
        return loss

    def forward(self, ca_flag=False):
        pos_h, pos_t, pos_r = self.get_postive_instance()
        neg_h, neg_t, neg_r = self.get_negtive_instance()

        p_h_e = self.ent_embeddings(pos_h) # batch, size
        p_t_e = self.ent_embeddings(pos_t)
        p_r_e = self.rel_embeddings(pos_r)
        n_h_e = self.ent_embeddings(neg_h)
        n_t_e = self.ent_embeddings(neg_t)
        n_r_e = self.rel_embeddings(neg_r)

        p_norm = self.norm_vector(pos_r)
        n_norm = self.norm_vector(neg_r)

        p_h = self._transfer(p_h_e, p_norm)
        p_t = self._transfer(p_t_e, p_norm)
        p_r = p_r_e
        n_h = self._transfer(n_h_e, n_norm)
        n_t = self._transfer(n_t_e, n_norm)
        n_r = n_r_e

        p_score = self._calc(p_h, p_t, p_r)
        n_score = self._calc(n_h, n_t, n_r)

        loss = self.loss_func(p_score, n_score)

        if self.regul_rate != 0 and not ca_flag:
            loss += self.regul_rate * self.regularization()

        return loss

    def regularization(self):
        pos_h, pos_t, pos_r = self.get_postive_instance()
        p_h_e = self.ent_embeddings(pos_h)  # batch, size
        p_t_e = self.ent_embeddings(pos_t)
        p_r_e = self.rel_embeddings(pos_r)
        p_r_norm = self.norm_vector(pos_r)
        regul = (torch.mean(p_h_e ** 2) +
                 torch.mean(p_t_e ** 2) +
                 torch.mean(p_r_e ** 2) +
                 torch.mean(p_r_norm ** 2)) / 4
        return regul

    def predict(self, predict_h, predict_t, predict_r):
        p_h_e = self.ent_embeddings(predict_h)
        p_t_e = self.ent_embeddings(predict_t)
        p_r_e = self.rel_embeddings(predict_r)
        p_norm = self.norm_vector(predict_r)

        p_h = self._transfer(p_h_e, p_norm)
        p_t = self._transfer(p_t_e, p_norm)
        p_r = p_r_e

        p_score = self._calc(p_h, p_t, p_r)

        return p_score.cpu()
