import torch
import torch.nn as nn
import torch.nn.functional as F
from models.Model import *


class RotatEMetaJoint(Model):
    def __init__(self, config, epsilon=2.0):
        super(RotatEMetaJoint, self).__init__(config)

        self.margin = self.config.margin
        self.epsilon = epsilon

        self.dim_e = self.ent_size * 2
        self.dim_r = self.rel_size

        self.ent_embeddings = nn.Embedding(self.entityTotal + 1, self.dim_e, padding_idx=0)
        self.rel_embeddings = nn.Embedding(self.propertyTotal + 1, self.dim_r, padding_idx=0)

        self.ent_embeddings.weight = nn.Parameter(
            torch.cat((torch.zeros(1, self.dim_e), torch.randn(self.entityTotal, self.dim_e)), 0))
        self.rel_embeddings.weight = nn.Parameter(
            torch.cat((torch.zeros(1, self.dim_r), torch.randn(self.propertyTotal, self.dim_r)), 0))

        self.ent_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_e]),
            requires_grad=False
        )

        self.rel_embedding_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.dim_r]),
            requires_grad=False
        )

        nn.init.uniform_(
            tensor=self.ent_embeddings.weight.data[1:],
            # a=-self.ent_embedding_range.item(),
            # b=self.ent_embedding_range.item()
        )

        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data[1:],
            # a=-self.rel_embedding_range.item(),
            # b=self.rel_embedding_range.item()
        )

        self.margin = nn.Parameter(torch.Tensor([self.config.margin]))
        self.criterion = nn.LogSigmoid()

        self.tar_att = nn.Parameter(torch.randn(1, self.dim_e, requires_grad=True))
        self.sour_att = nn.Parameter(torch.randn(1, self.dim_e,requires_grad=True))
        nn.init.xavier_uniform_(self.tar_att)
        nn.init.xavier_uniform_(self.sour_att)

        self.sigmoid = nn.Sigmoid()

    def loss_func(self, p_score, n_score):
        return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

    def _calc(self, h, t, r):
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0).sum(dim=-1)

        tar_att_vec = self.tar_att.expand(len(r), self.dim_e)
        tar_attr_re, tar_attr_im = torch.chunk(tar_att_vec, 2, dim=-1)
        ta_score_re = re_head * tar_attr_re - im_head * tar_attr_im
        ta_score_im = re_head * tar_attr_im + im_head * tar_attr_re
        ta_score_re = ta_score_re - re_relation
        ta_score_im = ta_score_im - im_relation
        ta_score = torch.stack([ta_score_re, ta_score_im], dim=0)
        if self.norm_flag:
            ta_score = F.normalize(ta_score, 2, -1)
        ta_score = ta_score.norm(dim=0).sum(-1)

        sour_att_vec = self.sour_att.expand(len(r), self.dim_e)
        sour_attr_re, sour_attr_im = torch.chunk(sour_att_vec, 2, dim=-1)
        sa_score_re = re_tail * sour_attr_re - im_tail * sour_attr_im
        sa_score_im = re_tail * sour_attr_im + im_tail * sour_attr_re
        sa_score_re = sa_score_re - re_relation
        sa_score_im = sa_score_im - im_relation
        sa_score = torch.stack([sa_score_re, sa_score_im], dim=0)
        if self.norm_flag:
            sa_score = F.normalize(sa_score, 2, -1)
        sa_score = sa_score.norm(dim=0).sum(dim=-1)
        # print('3score: ',score, '3ta: ',ta_score, '3sa: ', sa_score)
        score = score + ta_score + sa_score
        return score

    def _calc_ca(self, c, a):
        if self.norm_flag:
            c = F.normalize(c, 2, -1)
            a = F.normalize(a, 2, -1)

        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(c, 2, dim=-1)
        phase_relation = a / (self.rel_embedding_range.item() / pi)
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        tar_att_vec = self.tar_att.expand(len(a), self.dim_e)
        tar_attr_re, tar_attr_im = torch.chunk(tar_att_vec, 2, dim=-1)
        ta_score_re = re_head * tar_attr_re - im_head * tar_attr_im
        ta_score_im = re_head * tar_attr_im + im_head * tar_attr_re
        ta_score_re = ta_score_re - re_relation
        ta_score_im = ta_score_im - im_relation
        ta_score = torch.stack([ta_score_re, ta_score_im], dim=0)
        if self.norm_flag:
            ta_score = F.normalize(ta_score, 2, -1)
        ta_score = ta_score.norm(dim=0).sum(-1)

        sour_att_vec = self.sour_att.expand(len(c), self.dim_e)
        sour_attr_re, sour_attr_im = torch.chunk(sour_att_vec, 2, dim=-1)
        sa_score_re = re_head * sour_attr_re - im_head * sour_attr_im
        sa_score_im = re_head * sour_attr_im + im_head * sour_attr_re
        sa_score_re = sa_score_re - re_relation
        sa_score_im = sa_score_im - im_relation
        sa_score = torch.stack([sa_score_re, sa_score_im], dim=0)
        if self.norm_flag:
            sa_score = F.normalize(sa_score, 2, -1)
        sa_score = sa_score.norm(dim=0).sum(dim=-1)

        # print('2ta', ta_score,'2sa', sa_score)
        score = ta_score + sa_score

        return score

    def forward(self, ca_flag=False):

        if ca_flag:
            pos_c, pos_a = self.get_postive_instance_ca()
            neg_c, neg_a = self.get_negtive_instance_ca()

            p_c_e = self.ent_embeddings(pos_c)  # batch, size
            p_a_e = self.rel_embeddings(pos_a)
            n_c_e = self.ent_embeddings(neg_c)
            n_a_e = self.rel_embeddings(neg_a)

            p_score = self.margin - self._calc_ca(p_c_e, p_a_e)
            n_score = self.margin - self._calc_ca(n_c_e, n_a_e)

            loss = self.loss_func(p_score, n_score)
        else:
            pos_h, pos_t, pos_r = self.get_postive_instance()
            neg_h, neg_t, neg_r = self.get_negtive_instance()

            p_h_e = self.ent_embeddings(pos_h)  # batch, size
            p_t_e = self.ent_embeddings(pos_t)
            p_r_e = self.rel_embeddings(pos_r)
            n_h_e = self.ent_embeddings(neg_h)
            n_t_e = self.ent_embeddings(neg_t)
            n_r_e = self.rel_embeddings(neg_r)

            p_score = self.margin - self._calc(p_h_e, p_t_e, p_r_e)
            n_score = self.margin - self._calc(n_h_e, n_t_e, n_r_e)

            loss = self.loss_func(p_score, n_score)

        if self.regul_rate != 0 and not ca_flag:
            loss += self.regul_rate * self.regularization()

        return loss

    def predict(self, predict_h, predict_t, predict_r):
        p_h = self.ent_embeddings(predict_h)
        p_t = self.ent_embeddings(predict_t)
        p_r = self.rel_embeddings(predict_r)

        p_score = self._calc(p_h, p_t, p_r)

        return p_score.cpu()

    def regularization(self):
        pos_h, pos_t, pos_r = self.get_postive_instance()
        h = self.ent_embeddings(pos_h)  # batch, size
        t = self.ent_embeddings(pos_t)
        r = self.rel_embeddings(pos_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2)) / 3
        return regul
