import torch.nn as nn


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.ent_size = self.config.ent_size
        self.rel_size = self.config.rel_size
        self.batch_size = self.config.batch_size
        self.entity2id = self.config.entity2id
        self.property2id = self.config.property2id
        self.entityTotal = self.config.entityTotal
        self.propertyTotal = self.config.propertyTotal
        self.max_attr_size = self.config.max_attr_size
        self.matrix_rand_init = self.config.matrix_rand_init
        self.conid2attrid = self.config.conid2attrid
        self.p_norm = self.config.p_norm
        self.margin = self.config.margin
        self.regul_rate = self.config.regul_rate
        self.norm_flag = self.config.norm_flag
        self.use_cuda = self.config.use_cuda

    def get_postive_instance(self):
        self.postive_h = self.config.STbatch[:, 0, 0]
        self.postive_t = self.config.STbatch[:, 0, 1]
        self.postive_r = self.config.STbatch[:, 0, 2]
        if self.config.use_cuda:
            self.postive_h = self.postive_h.cuda()
            self.postive_t = self.postive_t.cuda()
            self.postive_r = self.postive_r.cuda()

        return self.postive_h, self.postive_t, self.postive_r

    def get_negtive_instance(self):
        self.negative_h = self.config.STbatch[:, 1, 0]
        self.negative_t = self.config.STbatch[:, 1, 1]
        self.negative_r = self.config.STbatch[:, 1, 2]
        if self.config.use_cuda:
            self.negative_h = self.negative_h.cuda()
            self.negative_t = self.negative_t.cuda()
            self.negative_r = self.negative_r.cuda()

        return self.negative_h, self.negative_t, self.negative_r

    def get_postive_instance_ca(self):
        self.postive_c = self.config.STbatch[:, 0, 0]
        self.postive_a = self.config.STbatch[:, 0, 1]
        if self.config.use_cuda:
            self.postive_c = self.postive_c.cuda()
            self.postive_a = self.postive_a.cuda()

        return self.postive_c, self.postive_a

    def get_negtive_instance_ca(self):
        self.negative_c = self.config.STbatch[:, 1, 0]
        self.negative_a = self.config.STbatch[:, 1, 1]
        if self.config.use_cuda:
            self.negative_c = self.negative_c.cuda()
            self.negative_a = self.negative_a.cuda()

        return self.negative_c, self.negative_a
