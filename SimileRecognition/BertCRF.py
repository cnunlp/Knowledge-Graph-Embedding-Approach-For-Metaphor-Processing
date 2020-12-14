from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
import torch.nn as nn
import torch


def argmax(v):
    _, idx = torch.max(v, 1)
    return idx.tolist()[0]


def log_sum_exp(batch_vec):
    res_list = []
    for vec in batch_vec:
        max_score = vec[argmax(vec.view(1, -1))]
        max_score_broadcast = max_score.expand(1, len(vec))
        results = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))
        res_list.append(results.view(1, -1))
    return torch.cat(res_list).view(-1, 1)  # (batch,1)


def log_sum_exp_v1(vec, dim=1):
    max_score, _ = torch.max(vec, dim=dim, keepdim=True)
    return max_score + \
        torch.logsumexp(vec - max_score, dim=dim, keepdim=True)


class BertCRF(BertPreTrainedModel):
    def __init__(self, config, num_labels, label_map):
        super(BertCRF, self).__init__(config)
        self.num_labels = num_labels
        self.label_map = label_map
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.hidden2tag = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)

        self.transitions = nn.Parameter(torch.randn(num_labels, self.num_labels))
        self.transitions.data[self.label_map['START'], :] = -10000
        self.transitions.data[:, self.label_map['STOP']] = -10000

    def _forward_alg(self, feats):
        feats_t = torch.transpose(feats, 0, 1)
        init_alphas = torch.full((len(feats), 1, self.num_labels), -10000)
        init_alphas[:, 0, self.label_map['START']] = 0
        forward_var = init_alphas.cuda()
        # forward_var = init_alphas
        for feat in feats_t:

            emit_score = feat.unsqueeze(-1).expand(len(feats), self.num_labels, self.num_labels)
            trans_score = self.transitions.expand(len(feats), self.num_labels, self.num_labels)
            next_tag_var = forward_var.expand(len(feats), self.num_labels, self.num_labels) + emit_score + trans_score
            forward_var = log_sum_exp_v1(next_tag_var, dim=2).view(len(feats), 1, -1)
        terminal_var = forward_var.squeeze(1) + self.transitions[self.label_map['STOP']]
        batch_alpha = log_sum_exp_v1(terminal_var)

        return batch_alpha

    def _sentence_score(self, batch_feats, batch_tags):
        batch_scores = []
        for (feats, tags) in zip(batch_feats, batch_tags):
            # start_tag = torch.tensor([self.label_map['START']], dtype=torch.long)
            start_tag = torch.tensor([self.label_map['START']], dtype=torch.long).cuda()
            tags = torch.cat((start_tag, tags))
            score = 0
            for i, feat in enumerate(feats):
                score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
            score = score + self.transitions[self.label_map['STOP'], tags[-1]]
            batch_scores.append(score.view(1, -1))
        return torch.cat(batch_scores).view(len(batch_feats), 1)

    def _get_bert_feats(self, input_ids, token_type_ids=None, attention_mask=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        feats = self.hidden2tag(sequence_output)
        return feats

    def _neg_likelihood_loss(self, input_ids, label_ids, segment_ids, input_mask):
        bert_feats = self._get_bert_feats(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        forward_score = self._forward_alg(bert_feats)
        gold_score = self._sentence_score(bert_feats, label_ids)
        return torch.sum(forward_score - gold_score) / len(input_ids)

    def _viterbi_decode(self, feats):
        backpointers = []
        feats_t = torch.transpose(feats, 0, 1)  # (hidden_size, batch, tag_size)
        init_vvars = torch.full((len(feats), self.num_labels), -10000.)
        init_vvars[:, self.label_map['START']] = 0
        # forward_var = init_vvars.cuda()
        forward_var = init_vvars
        for feat in feats_t:  # feat: (batch, tag_size)
            bptrs_t = []  # stories best tag's ids for each of feats then append to backpointers
            viterbivars_t = []  # stories best tag's scores for each of feats then equals forward_var
            for next_tag in range(self.num_labels):
                next_tag_var = forward_var + self.transitions[next_tag].view(1, -1).expand(len(feats),
                                                                                           self.num_labels)  # (batch,tag_size)
                best_tag_var, best_tag_id = torch.max(next_tag_var, 1)
                bptrs_t.append(best_tag_id.view(len(feats), -1))
                viterbivars_t.append(best_tag_var.view(len(feats), -1))
            forward_var = torch.cat(viterbivars_t, 1) + feat # (batch, tag_size)
            backpointers.append(torch.cat(bptrs_t, 1))  # [(batch, tag_size),(batch, tag_size)...]
        terminal_var = forward_var + self.transitions[self.label_map['STOP']].expand(len(feats),
                                                                                  self.num_labels)  # (batch, tag_size)
        best_tag_var, best_tag_id = torch.max(terminal_var, 1)  # (batch, 1)
        path_score = best_tag_var  # (batch, 1)
        best_path = [best_tag_id.view(len(feats), -1)]
        for bptrs_t in reversed(backpointers):
            temp_list = []
            for i, bptr in enumerate(bptrs_t):
                temp_list.append(bptr[best_tag_id[i]].view(1))
            best_tag_id = torch.cat(temp_list)
            best_path.append(best_tag_id.view(len(feats), -1))
        best_path = torch.cat(best_path, 1)
        batch_best_path = []
        for path in best_path:
            path = path.tolist()
            start = path.pop()
            assert start == self.label_map['START']  # Sanity check
            path.reverse()
            batch_best_path.append(path)
        return path_score, batch_best_path

    def forward(self, input_ids, segment_ids, input_mask):
        bert_feats = self._get_bert_feats(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)
        batch_path_socre, batch_tag_seq = self._viterbi_decode(bert_feats)

        return batch_tag_seq
