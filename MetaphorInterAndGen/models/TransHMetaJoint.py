from models.Model import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransHMetaJoint(Model):
    def __init__(self, config):
        super(TransHMetaJoint, self).__init__(config)
        self.ent_embeddings = nn.Embedding(self.entityTotal+1, self.ent_size, padding_idx = 0)
        self.rel_embeddings = nn.Embedding(self.propertyTotal+1, self.rel_size, padding_idx=0)
        self.norm_vector = nn.Embedding(self.propertyTotal+1, self.rel_size, padding_idx=0)

        self.tar_sim_matrix = nn.Parameter(torch.randn(self.rel_size, self.rel_size, requires_grad=True))
        self.sour_sim_matrix = nn.Parameter(torch.randn(self.rel_size, self.rel_size, requires_grad=True))
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

        self.tar_sim_matrix = nn.Parameter(torch.randn(self.rel_size, self.rel_size, requires_grad=True))
        nn.init.xavier_uniform_(self.tar_sim_matrix)
        self.sour_sim_matrix = nn.Parameter(torch.randn(self.rel_size, self.rel_size, requires_grad=True))
        nn.init.xavier_uniform_(self.sour_sim_matrix)

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p=2, dim=-1)
        return e - torch.sum(e * norm, -1, True) * norm

    def _calc(self, h, t, r):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        tri_score = torch.norm((h + r - t), self.p_norm, dim=-1)  # batch
        ta_score = torch.sum(torch.matmul(h, self.tar_sim_matrix) * r, -1)
        sa_score = torch.sum(torch.matmul(t, self.sour_sim_matrix) * r, -1)
        score = tri_score - ta_score - sa_score

        return score

    def _calc_ca(self, c, a, pn_flag):
        if self.norm_flag:
            c = F.normalize(c, 2, -1)
            a = F.normalize(a, 2, -1)
        ta_score = torch.sum(torch.matmul(c, self.tar_sim_matrix) * a, -1)  # batch
        sa_score = torch.sum(torch.matmul(c, self.sour_sim_matrix) * a, -1) # batch
        if pn_flag == 'p':
            score = self.sigmoid(ta_score+sa_score)
        elif pn_flag == 'n':
            score = self.sigmoid(-(ta_score+sa_score))
        score = torch.log(score)

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
        if ca_flag:
            pos_c, pos_a = self.get_postive_instance_ca()
            neg_c, neg_a = self.get_negtive_instance_ca()

            p_c_e = self.ent_embeddings(pos_c)  # batch, size
            p_a_e = self.rel_embeddings(pos_a)
            n_c_e = self.ent_embeddings(neg_c)
            n_a_e = self.rel_embeddings(neg_a)

            p_norm = self.norm_vector(pos_a)
            n_norm = self.norm_vector(neg_a)

            p_c = self._transfer(p_c_e, p_norm)
            p_a = p_a_e
            n_c = self._transfer(n_c_e, n_norm)
            n_a = n_a_e

            p_score = self._calc_ca(p_c, p_a, 'p')
            n_score = self._calc_ca(n_c, n_a, 'n')

            loss = torch.sum(- p_score - n_score)
        else:
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
