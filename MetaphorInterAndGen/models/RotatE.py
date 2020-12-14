import torch
import torch.nn as nn
import torch.nn.functional as F
from Model import *


class RotatE(Model):
    def __init__(self, config, epsilon=2.0):
        super(RotatE, self).__init__(config)

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
        )

        nn.init.uniform_(
            tensor=self.rel_embeddings.weight.data[1:],
        )

        self.margin = nn.Parameter(torch.Tensor([self.config.margin]))
        self.criterion = nn.LogSigmoid()

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

        return score

    def forward(self, ca_flag=False):
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
