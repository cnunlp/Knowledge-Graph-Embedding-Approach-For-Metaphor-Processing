import numpy as np
import torch
from random import uniform, sample, randint
from collections import Counter
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
    return data2id, id2data, entid2tags


# generate data2id dic
def generate_data_idx(file_path, entity2id, property2id):
    data2id = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) > 1:
                line_parts = line.strip().split()
                data2id.append([entity2id[line_parts[0]], entity2id[line_parts[1]], property2id[line_parts[2]]])
    return data2id


def generate_conceptid_to_attributesid(file_path, entity2id, property2id, max_attr_size):
    ca2id_dic = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 1:
                line_parts = line.strip().split(' ')
                concept = line_parts[0]
                attr = line_parts[1]
                ca2id_dic.setdefault(entity2id[concept], set()).add(property2id[attr])
    f.close()
    conid2Attrid = torch.zeros(len(entity2id) + 1, max_attr_size).long()
    for con in ca2id_dic.keys():
        attrs = list(ca2id_dic[con])
        attrs = attrs + [0] * (max_attr_size - len(attrs))
        conid2Attrid[con] = torch.LongTensor(attrs)
    return conid2Attrid


def generate_concept_attributes_idx(file_path, entity2id, property2id):
    data2id_dic = {}
    data2id = []
    conAttr2id_set = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) > 1:
                line_parts = line.strip().split()
                concept, pro, fre = line_parts[0], line_parts[1], line_parts[2]
                data2id_dic.setdefault(concept, []).append(' '.join([pro, fre]))
                conAttr2id_set.add(' '.join([str(entity2id[concept]), str(property2id[pro])]))
    f.close()
    for tar in data2id_dic.keys():
        tripletes = []
        weights = []
        for pf in data2id_dic[tar]:
            source = entity2id[tar]
            pro = property2id[pf.split()[0]]
            fre = int(pf.split()[1])
            tsp = [source, pro]
            tripletes.append(tsp)
            weights.append(fre)
        data2id.append([tripletes, weights])
    return data2id, conAttr2id_set


def weight_choice(weight):
    t = randint(0, sum(weight) - 1)
    for i, val in enumerate(weight):
        t -= val
        if t < 0:
            return i


def conAttr_choice(data2id):
    CA = []
    for t_w in data2id:
        triplets = t_w[0]
        weights = t_w[1]
        choice_t = triplets[weight_choice(weights)]
        CA.append(choice_t)
    return CA


# load hyponymy words for entities
def generate_entity_tyc_idx(file_path, entity2id):
    entid2tycid = {}
    hyponymy_file = open(file_path, 'r', encoding='utf-8')
    for line in hyponymy_file:
        line_parts = line.strip().split(' ')
        entity = entity2id[line_parts[0]]
        tycs = line_parts[1:]
        for t_word in tycs:
            t_word_id = entity2id[t_word]
            entid2tycid.setdefault(entity, []).append(t_word_id)
    return entid2tycid


# load embeds
def load_embeddings(word2id, embedding_path, n_words, embed_size):
    embedding_matrix = np.zeros((n_words + 1, embed_size))
    with open(embedding_path, "r", encoding="utf-8") as f:
        for line in f:
            line_parts = line.strip().split()
            if len(line_parts) <= 2:
                continue
            word = line_parts[0]
            if word in word2id:
                word_id = word2id[word]
                try:
                    embedding_matrix[word_id] = line_parts[1:]
                except:
                    print(word)

    return torch.from_numpy(embedding_matrix).squeeze().float()


# group train data
def get_batches(train2id, batch_size):
    batches = []
    for i in range(0, len(train2id), batch_size):
        if i + batch_size < len(train2id):
            batch = list(map(tuple, train2id[i:i + batch_size]))
        else:
            batch = list(map(tuple, train2id[i:]))
        batches.append(batch)
    return batches


def get_samples(data, size):
    return sample(data, size)


# get (S,S')
def get_tuples(Sbatch, sample_ent_cand_ids, sample_attr_cand_ids, conid2Attrid, conAttr2id_set, train2id_set,
               ca_flag=False):
    Tbatch = []
    for sbatch in Sbatch:
        if ca_flag:
            nbatch = getCorruptedTriplet_for_CA(sbatch, sample_ent_cand_ids, conAttr2id_set)
        else:
            nbatch = getCorruptedTriplet(sbatch, sample_ent_cand_ids, sample_attr_cand_ids, conid2Attrid, train2id_set)
        sample = (sbatch, nbatch)
        Tbatch.append(sample)

    Tbatch = torch.LongTensor(list(Tbatch))  # (batch, 2, 3)
    return Tbatch


# get adversary (S,S')
def get_adversary_tuples(Sbatch, hyponymy_words2id):
    Tbatch = set()
    for sbatch in Sbatch:
        self_against_sample = (sbatch, get_against_sample(sbatch))
        tyc_against_sample = (sbatch, get_tyc_sample(sbatch, hyponymy_words2id))
        Tbatch.add(self_against_sample)
        Tbatch.add(tyc_against_sample)
    Tbatch = torch.LongTensor(list(Tbatch))
    return Tbatch


def get_tyc_sample(triplet, entid2tycid):
    i = uniform(0, 2)
    concept = triplet[0] if 0 < i < 1 else triplet[1]
    tyc_ids = entid2tycid[concept]
    tyc_id = get_samples(tyc_ids, 1)[0]
    corruptedTriplet = (concept, tyc_id, triplet[2]) if 0 < i < 1 else (tyc_id, concept, triplet[2])

    return corruptedTriplet


def get_against_sample(triplet):
    i = uniform(0, 2)
    if 0 < i < 1:
        corruptedTriplet = (triplet[0], triplet[0], triplet[2])
    else:
        corruptedTriplet = (triplet[1], triplet[1], triplet[2])
    return corruptedTriplet


# generate corrupted triplet for training
def getCorruptedTriplet(triplet, sample_ent_cand_ids, sample_attr_cand_ids, conid2Attrid, train2id_set):
    i = uniform(0, 3)
    if 0 < i < 1:
        while True:
            # 保证替换的t'的属性集合中不包含共有属性a
            entityTemp = get_samples(sample_ent_cand_ids, 1)[0]
            entTmp_a = set(conid2Attrid[entityTemp].view(-1).tolist())
            tri_str = ' '.join(map(str, [entityTemp, triplet[1], triplet[2]]))
            if entityTemp != triplet[0] and triplet[2] not in entTmp_a and tri_str not in train2id_set:
                break
        corruptedTriplet = (entityTemp, triplet[1], triplet[2])
    elif 1 < i < 2:
        while True:
            # 保证替换的s’的属性集合中不包含共有属性a
            entityTemp = get_samples(sample_ent_cand_ids, 1)[0]
            entTmp_a = set(conid2Attrid[entityTemp].view(-1).tolist())
            tri_str = ' '.join(map(str, [triplet[0], entityTemp, triplet[2]]))
            if entityTemp != triplet[1] and triplet[2] not in entTmp_a and tri_str not in train2id_set:
                break
        corruptedTriplet = (triplet[0], entityTemp, triplet[2])
    else:
        while True:
            # 保证替换的a不在t_a和s_a的交集中
            propertyTemp = get_samples(sample_attr_cand_ids, 1)[0]
            h_a = set(conid2Attrid[torch.LongTensor([triplet[0]])].view(-1).tolist())
            t_a = set(conid2Attrid[torch.LongTensor([triplet[1]])].view(-1).tolist())
            ht_a = h_a | t_a
            tri_str = ' '.join(map(str, [triplet[0], triplet[1], propertyTemp]))
            if propertyTemp != triplet[2] and propertyTemp not in ht_a and tri_str not in train2id_set:
                break
        corruptedTriplet = (triplet[0], triplet[1], propertyTemp)

    return corruptedTriplet


def getCorruptedTriplet_for_CA(triplet, sample_ent_cand_ids, conAttr2id_set):
    while True:
        # 保证替换的c'和属性a未共现过
        entityTemp = get_samples(sample_ent_cand_ids, 1)[0]
        tmpTrip = ' '.join([str(entityTemp), str(triplet[1])])
        if entityTemp != triplet[0] and tmpTrip not in conAttr2id_set:
            break
    corruptedTriplet = (entityTemp, triplet[1])

    return corruptedTriplet


# generate corrupted triplet for testing
def get_test_candidates(triplet, entity_candidate_ids, property_candidate_ids, flag):
    if flag == "p":
        corruptedRelList = set()
        for p in property_candidate_ids:
            corruptedRel = (triplet[0], triplet[1], p)
            if corruptedRel not in corruptedRelList:
                corruptedRelList.add(corruptedRel)
        corruptedRelList = torch.LongTensor(list(corruptedRelList))
        return corruptedRelList

    elif flag == "t":
        corruptedTailList = set()
        for e in entity_candidate_ids:
            corruptedTail = (triplet[0], e, triplet[2])
            if e != triplet[0] and corruptedTail not in corruptedTailList:
                corruptedTailList.add(corruptedTail)
            else:
                continue
        corruptedTailList = torch.LongTensor(list(corruptedTailList))
        return corruptedTailList


def read_sample_candidates(path, word2id):
    candi = []
    data = open(path, 'r', encoding='utf-8')
    for line in data:
        w = line.strip().split(' ')[0]
        candi.append(word2id[w])
    return candi


def generate_ca_idx(file_path, entity2id, property2id):
    data2id = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if len(line.strip()) > 1:
                line_parts = line.strip().split()
                data2id.append([entity2id[line_parts[0]], property2id[line_parts[1]], int(line_parts[2])])
    return data2id


def get_attr_candi(target, source, con_attrs):
    t_a = con_attrs[target].keys()
    s_a = con_attrs[source].keys()
    t_s_a = t_a & s_a
    candis = set()
    for a in t_s_a:
        candis.add((target, source, a))
    attr_candi = torch.LongTensor(list(candis))

    return attr_candi


def get_sour_candi(target, attr, entities, con2attrs):
    candi = set()
    for ent in entities:
        ent_pros = con2attrs[ent].keys()
        if attr in ent_pros and ent != target:
            candi.add((target, ent, attr))
    candis = torch.LongTensor(list(candi))
    return candis


# link prediction for head, tail ,property
def link_prediction_with_cilin_tag(test2id, model, id2entity, id2property, entity_candidate_ids, property_candidate_ids, entid2tags, proid2tags, flag, use_cuda):
    mean_rank = []
    MRR = []
    hits_1_counts = 0
    hits_10_counts = 0
    mean_rank_tag = []
    MRR_tag = []
    hits_1_counts_tag = 0
    hits_10_counts_tag = 0
    for triplet in test2id:
        score_dic = {}
        test_triplet_list = get_test_candidates(triplet, entity_candidate_ids, property_candidate_ids, flag)
        if use_cuda:
            test_triplet_list = test_triplet_list.cuda()

        for t in test_triplet_list:
            t = t.view(len(t), 1)
            score = model.predict(t[0], t[1], t[2])
            score_dic[t] = score.data
        score_list = Counter(score_dic).most_common()
        score_list.reverse()

        if flag == 'p':
            print("labeled_triplet: ", (id2entity[triplet[0]], id2entity[triplet[1]], id2property[triplet[2]]))
            print('labeled_attr_tag: ', ' '.join(proid2tags[triplet[2]]))
            print('predict_top10:', end=' ')
            for t in score_list[:10]:
                pre_attr = id2property[t[0][2].item()]
                pre_attr_tag = proid2tags[t[0][2].item()]
                print(pre_attr + '_' + '_'.join(pre_attr_tag), end=' ')
            print('\n', end='')

            # without tag
            for i, t in enumerate(score_list):
                if t[0][2][0] == triplet[2]:
                    print("The rank of true attribute: ", i + 1)
                    mean_rank.append(i + 1)
                    MRR.append(1/(i+1))
                    if (i + 1) == 1:
                        hits_1_counts += 1
                    if (i + 1) <= 10:
                        hits_10_counts += 1
                    break

            # with tags
            pred = False
            for j, tr in enumerate(score_list):
                pred_tags = proid2tags[tr[0][2][0].item()]
                for ltag in proid2tags[triplet[2]]:
                    if ltag in pred_tags:
                        pred = True
                        break
                if pred:
                    print("The rank of true attribute tag: ", j + 1)
                    MRR_tag.append(1/(j+1))
                    mean_rank_tag.append(j + 1)
                    if (j + 1) == 1:
                        hits_1_counts_tag += 1
                    if (j + 1) <= 10:
                        hits_10_counts_tag += 1
                    break

        elif flag == 't':
            print("labeled_triplet: ", (id2entity[triplet[0]], id2entity[triplet[1]], id2property[triplet[2]]))
            print('labeled_source_tag: ', ' '.join(entid2tags[triplet[1]]))
            print('predict_top10:', end=' ')
            for t in score_list[:10]:
                pre_sour = id2entity[t[0][1].item()]
                pre_sour_tag = entid2tags[t[0][1].item()]
                print(pre_sour + '_' + '_'.join(pre_sour_tag), end=' ')
            print('\n', end='')

            # without tags
            for i, t in enumerate(score_list):
                if t[0][1][0] == triplet[1]:
                    print("The rank of true source: ", i + 1)
                    mean_rank.append(i + 1)
                    MRR.append(1/(i+1))
                    if (i + 1) == 1:
                        hits_1_counts += 1
                    if (i + 1) <= 10:
                        hits_10_counts += 1
                    break
            # with tags
            pred = False
            for j, tr in enumerate(score_list):
                pred_tags = entid2tags[tr[0][1][0].item()]
                for ltag in entid2tags[triplet[1]]:
                    if ltag in pred_tags:
                        pred = True
                        break
                if pred:
                    print("The rank of true source tag: ", j + 1)
                    MRR_tag.append(1/(j+1))
                    mean_rank_tag.append(j + 1)
                    if (j+1) == 1:
                        hits_1_counts_tag += 1
                    if (j + 1) <= 10:
                        hits_10_counts_tag += 1
                    break

    print("The mean rank is: ", sum(mean_rank) / len(mean_rank))
    print("The Hits 10%% is: %f%%" % ((hits_10_counts / len(mean_rank)) * 100))
    print("The mean rank with tag is: ", sum(mean_rank_tag) / len(mean_rank_tag))
    print("The Hits 10%% with tag is: %f%%" % ((hits_10_counts_tag / len(mean_rank_tag)) * 100))

    print("The MRR is: ", sum(MRR) / len(MRR))
    print("The Hits 1%% is: %f%%" % ((hits_1_counts / len(MRR)) * 100))
    print("The MRR with tag is: ", sum(MRR_tag) / len(MRR_tag))
    print("The Hits 1%% with tag is: %f%%" % ((hits_1_counts_tag / len(MRR_tag)) * 100))


def max_min_normalize(x, max, min):
    if x > max:
        x = max
    if x < min:
        x = min
    x = (x - min) / (max - min)
    return x


def get_max_min_score(train_triplets, model):
    score_list = []
    for t in train_triplets:
        t = t.view(-1, 1).cuda()
        score = model.predict(t[0], t[1], t[2])
        score_list.append(score)
    max_s = max(score_list)
    min_s = min(score_list)
    return max_s, min_s


def triplet_classification(triplets, model, threshold, max_s, min_s):
    score_list = []
    for t in triplets:
        t = t.view(-1, 1).cuda()
        score = model.predict(t[0], t[1], t[2])
        score_list.append(score)
    scores = [max_min_normalize(s, max_s, min_s) for s in score_list]
    pre_label = [1 if s < threshold else 0 for s in scores]
    return pre_label