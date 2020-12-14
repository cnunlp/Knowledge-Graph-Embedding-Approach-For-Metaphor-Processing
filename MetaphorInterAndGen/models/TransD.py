import torch
import torch.nn as nn
from Model import *
import torch.nn.functional as F


class TransD(Model):
    def __init__(self, config):
        super(TransD, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.entityTotal+1, self.ent_size, padding_idx = 0)
        self.rel_embeddings = nn.Embedding(self.propertyTotal+1, self.rel_size, padding_idx=0)
        self.ent_transfer = nn.Embedding(self.entityTotal + 1, self.ent_size, padding_idx=0)
        self.rel_transfer = nn.Embedding(self.propertyTotal+1, self.rel_size, padding_idx=0)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        self.ent_embeddings.weight = nn.Parameter(torch.cat((torch.zeros(1, self.ent_size), torch.randn(self.entityTotal, self.ent_size)), 0))
        self.rel_embeddings.weight = nn.Parameter(torch.cat((torch.zeros(1, self.rel_size), torch.randn(self.propertyTotal, self.rel_size)), 0))
        self.ent_transfer.weight = nn.Parameter(torch.cat((torch.zeros(1, self.ent_size), torch.randn(self.entityTotal, self.ent_size)), 0))
        self.rel_transfer.weight = nn.Parameter(torch.cat((torch.zeros(1, self.rel_size), torch.randn(self.propertyTotal, self.rel_size)), 0))

        nn.init.xavier_uniform_(self.ent_embeddings.weight.data[1:, ])
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data[1:, ])
        nn.init.xavier_uniform_(self.ent_transfer.weight.data[1:, ])
        nn.init.xavier_uniform_(self.rel_transfer.weight.data[1:, ])

    def _transfer(self, e, e_transfer, r_transfer):
        return F.normalize(
                e + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
        )

    def _calc(self, h, t, r):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)

        score = torch.norm((h + r - t), self.p_norm, dim=-1)  # batch

        return score

    def loss_func(self, p_score, n_score):
        criterion = nn.MarginRankingLoss(self.margin, False)
        y = torch.Tensor([-1])
        if self.use_cuda:
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

        p_h_t = self.ent_transfer(pos_h) # batch, size
        p_t_t = self.ent_transfer(pos_t)
        p_r_t = self.rel_transfer(pos_r)
        n_h_t = self.ent_transfer(neg_h)
        n_t_t = self.ent_transfer(neg_t)
        n_r_t = self.rel_transfer(neg_r)

        p_h = self._transfer(p_h_e, p_h_t, p_r_t)  # batch_size
        p_t = self._transfer(p_t_e, p_t_t, p_r_t)
        p_r = p_r_e
        n_h = self._transfer(n_h_e, n_h_t, n_r_t)
        n_t = self._transfer(n_t_e, n_t_t, n_r_t)
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
        p_h_t = self.ent_transfer(pos_h)  # batch, size
        p_t_t = self.ent_transfer(pos_t)
        p_r_t = self.rel_transfer(pos_r)

        regul = (torch.mean(p_h_e ** 2) +
                 torch.mean(p_t_e ** 2) +
                 torch.mean(p_r_e ** 2) +
                 torch.mean(p_h_t ** 2) +
                 torch.mean(p_t_t ** 2) +
                 torch.mean(p_r_t ** 2))/ 6
        return regul

    def predict(self, predict_h, predict_t, predict_r):
        p_h_e = self.ent_embeddings(predict_h)
        p_t_e = self.ent_embeddings(predict_t)
        p_r_e = self.rel_embeddings(predict_r)

        p_h_t = self.ent_transfer(predict_h) # batch, size
        p_t_t = self.ent_transfer(predict_t)
        p_r_t = self.rel_transfer(predict_r)

        p_h = self._transfer(p_h_e, p_h_t, p_r_t)  # batch_size
        p_t = self._transfer(p_t_e, p_t_t, p_r_t)
        p_r = p_r_e

        p_score = self._calc(p_h, p_t, p_r)

        return p_score.cpu()
