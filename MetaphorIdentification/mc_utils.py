import torch
from collections import OrderedDict

# generate entity2id,property2id dic
def generate_entity_property_idx(file_path):
    data2id = OrderedDict()
    entid2tags = {}
    data_file = open(file_path, 'r', encoding='utf-8')
    for line in data_file:
        line_parts = line.strip().split(' ')
        word = line_parts[0]
        tag = line_parts[1]
        tags = tag.split('_') if '_' in tag else [tag]
        if word not in data2id.keys():
            data2id[word] = len(data2id) + 1
            entid2tags[data2id[word]] = tags
    id2data = dict(zip(data2id.values(), data2id.keys()))
    return data2id, id2data


# generate data2id dic
def generate_data_idx(file_path, entity2id, label2id):
    data2id = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) > 1:
                line_parts = line.strip().split()
                # if line_parts[-1] != '-1':
                data2id.append([entity2id[line_parts[0]], entity2id[line_parts[1]], label2id[line_parts[-1]]])

    return torch.LongTensor(data2id)


# load trans embeddings
def load_trans_embeddings(ent_embeds_path, rel_embeds_path):
    pre_ent_embeds = torch.load(ent_embeds_path, map_location=lambda storage, loc: storage)['ent_embeddings.weight']
    pre_rel_embeds = torch.load(rel_embeds_path, map_location=lambda storage, loc: storage)['rel_embeddings.weight']

    return pre_ent_embeds, pre_rel_embeds


# load tencent embeddings
def load_ten_embeddings(file_path, entity2id, property2id, embed_size):
    entity_embedding_matrix = torch.zeros(len(entity2id)+1, embed_size)
    pro_embedding_matrix = torch.zeros(len(property2id)+1, embed_size)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line_parts = line.strip().split(' ')
            if len(line_parts) <= 2:
                continue
            word = line_parts[0]
            if word in entity2id.keys():
                entity_id = entity2id[word]
                entity_embedding_matrix[entity_id] = torch.Tensor(list(map(float, line_parts[1:])))
            elif word in property2id.keys():
                pro_id = property2id[word]
                pro_embedding_matrix[pro_id] = torch.Tensor(list(map(float, line_parts[1:])))

    return entity_embedding_matrix, pro_embedding_matrix


def get_batches(data2id, batch_size):
    batches = []
    for i in range(0, len(data2id), batch_size):
        if i + batch_size < len(data2id):
            batch = data2id[i:i + batch_size, :]
        else:
            batch = data2id[i:, :]
        batches.append(batch)
    return batches
