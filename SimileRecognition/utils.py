import torch


def collate_fun(data):
    sent_len = [sent[0].tolist().index(0) for sent in data]
    max_len = max(sent_len)

    new_sents = []
    new_input_mask = []
    new_seg_id = []
    new_label_ids = []
    for sent in data:
        new_sents.append(sent[0][:max_len])
        new_input_mask.append(sent[1][:max_len])
        new_seg_id.append(sent[2][:max_len])
        new_label_ids.append(sent[3][:max_len])
    new_data = (torch.cat(new_sents).view(len(data), -1), torch.cat(new_input_mask).view(len(data), -1), torch.cat(new_seg_id).view(len(data), -1), torch.cat(new_label_ids).view(len(data), -1))

    return new_data


def collate_fun_joint(data):
    sent_len = [sent[0].tolist().index(0) for sent in data]
    max_len = max(sent_len)

    new_sents = []
    new_input_mask = []
    new_seg_id = []
    new_label_ids = []
    new_sent_label_ids = []
    for sent in data:
        new_sents.append(sent[0][:max_len])
        new_input_mask.append(sent[1][:max_len])
        new_seg_id.append(sent[2][:max_len])
        new_label_ids.append(sent[3][:max_len])
        new_sent_label_ids.append(sent[4].view(-1))
    new_data = (torch.cat(new_sents).view(len(data), -1), torch.cat(new_input_mask).view(len(data), -1),
                torch.cat(new_seg_id).view(len(data), -1), torch.cat(new_label_ids).view(len(data), -1),
                torch.cat(new_sent_label_ids).view(-1))

    return new_data

