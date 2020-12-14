import torch
import torch.nn as nn


class SSN_model(nn.Module):
    def __init__(self, entity2id, property2id, pre_ent_embeds, pre_rel_embeds, args):
        super(SSN_model, self).__init__()
        self.pre_ent_embeds = pre_ent_embeds
        self.pre_rel_embeds = pre_rel_embeds
        self.args = args
        self.ent_embedding = nn.Embedding(len(entity2id)+1, args.embed_size)
        self.rel_embedding = nn.Embedding(len(property2id)+1, args.embed_size)

        if self.args.embed_flag == 'concat':
            self.W_g = nn.Parameter(torch.randn(args.embed_size*2, args.embed_size*2))
            self.W_z1 = nn.Parameter(torch.randn(args.embed_size*2, 128))
            self.W_z2 = nn.Parameter(torch.randn(args.embed_size*2, 128))
        else:
            self.W_g = nn.Parameter(torch.randn(args.embed_size, args.embed_size))
            self.W_z1 = nn.Parameter(torch.randn(args.embed_size, 128))
            self.W_z2 = nn.Parameter(torch.randn(args.embed_size, 128))
        self.W_d = nn.Parameter(torch.randn(128, 50))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.W_y = nn.Parameter(torch.randn(50, 1))

        self.init_weight()

    def init_weight(self):
        self.ent_embedding.weight = nn.Parameter(self.pre_ent_embeds)
        self.ent_embedding.weight.requires_grad = False
        self.rel_embedding.weight = nn.Parameter(self.pre_rel_embeds)
        self.rel_embedding.weight.requires_grad = False

        nn.init.xavier_uniform_(self.W_g.data)
        nn.init.xavier_uniform_(self.W_z1.data)
        nn.init.xavier_uniform_(self.W_z2.data)
        nn.init.xavier_uniform_(self.W_d.data)
        nn.init.xavier_uniform_(self.W_y.data)

    def _cal_loss(self, outputs,target):
        criterion = nn.MSELoss()
        if self.args.use_cuda:
            criterion = criterion.cuda()
        loss = criterion(outputs, target)
        return loss

    def forward(self, train_batch):
        h = train_batch[:, 0]
        t = train_batch[:, 1]
        target = train_batch[:, -1]
        h_e = self.ent_embedding(h)
        t_e = self.ent_embedding(t)
        g = self.sigmoid(torch.matmul(h_e, self.W_g))  # batch, embed
        x2 = t_e * g  # batch, embed
        z1 = self.tanh(torch.matmul(h_e, self.W_z1)) # batch, 128
        z2 = self.tanh(torch.matmul(x2, self.W_z2)) # batch, 128
        m = z1*z2 # batch, 128
        d = self.tanh(torch.matmul(m, self.W_d)) # batch, 50
        y = self.sigmoid(torch.matmul(d, self.W_y)).view(-1) # batch, 1
        loss = self._cal_loss(y, target.float())

        return loss

    def test(self, test_batch):
        h = test_batch[:, 0]
        t = test_batch[:, 1]
        target = test_batch[:, -1]
        h_e = self.ent_embedding(h)
        t_e = self.ent_embedding(t)
        g = self.sigmoid(torch.matmul(h_e, self.W_g))  # batch, embed
        x2 = t_e * g  # batch, embed
        z1 = self.tanh(torch.matmul(h_e, self.W_z1)) # batch, 128
        z2 = self.tanh(torch.matmul(x2, self.W_z2)) # batch, 128
        m = z1*z2 # batch, 128
        d = self.tanh(torch.matmul(m, self.W_d)) # batch, 50
        y = self.sigmoid(torch.matmul(d, self.W_y)).view(-1) # batch, 1
        loss = self._cal_loss(y, target.float())
        pre_labels = torch.zeros(len(test_batch)).int()
        for i, v in enumerate(y):
            pre_labels[i] = 1 if v > 0.8 else 0

        return loss, pre_labels.cpu()








