import torch
import torch.nn as nn


class MLP_model(nn.Module):
    def __init__(self, entity2id, property2id, pre_ent_embeds, pre_rel_embeds, args):
        super(MLP_model, self).__init__()
        self.pre_ent_embeds = pre_ent_embeds
        self.pre_rel_embeds = pre_rel_embeds
        self.args = args
        self.ent_embedding = nn.Embedding(len(entity2id), args.embed_size)
        self.rel_embedding = nn.Embedding(len(property2id), args.embed_size)
        if self.args.embed_flag == 'concat':
            self.mlp = nn.Sequential(
                nn.Linear(args.embed_size * 2 * 2, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, int(args.hidden_size / 2)),
                nn.ReLU(),
                nn.Linear(int(args.hidden_size / 2), 2),
                nn.Softmax(dim=1)
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(args.embed_size * 2, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size, int(args.hidden_size / 2)),
                nn.ReLU(),
                nn.Linear(int(args.hidden_size / 2), 2),
                nn.Softmax(dim=1)
            )
        self.init_weight()

    def init_weight(self):
        self.ent_embedding.weight = nn.Parameter(self.pre_ent_embeds)
        self.ent_embedding.weight.requires_grad = False
        self.rel_embedding.weight = nn.Parameter(self.pre_rel_embeds)
        self.rel_embedding.weight.requires_grad = False

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight.data)

    def _cal_loss(self, outputs,target):
        criterion = nn.NLLLoss()
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
        inputs = torch.cat((h_e, t_e), 1)
        outputs = self.mlp(inputs)
        loss = self._cal_loss(torch.log(outputs), target)

        return loss

    def test(self, test_batch):
        h = test_batch[:, 0]
        t = test_batch[:, 1]
        target = test_batch[:, -1]
        h_e = self.ent_embedding(h)
        t_e = self.ent_embedding(t)
        inputs = torch.cat((h_e, t_e), 1)
        outputs = self.mlp(inputs)
        loss = self._cal_loss(torch.log(outputs), target)
        max_value, pre_labels = torch.max(outputs, 1)

        return loss, pre_labels.cpu()








