import torch
import mc_utils
import torch.optim as optim
import os
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import logging
import argparse
from models.MLPModel import MLP_model
from models.SSNModel import SSN_model
import random

logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

random.seed(1)
torch.manual_seed(1)

class Classify(object):
    def __init__(self, args):
        super(Classify, self).__init__()
        self.args = args
        self.train_path = os.path.join(args.data_dir, 'train.txt')
        self.dev_path = os.path.join(args.data_dir, 'test.txt')
        self.test_path = os.path.join(args.data_dir, 'test.txt')
        self.entity_path = os.path.join(args.data_dir, 'entity_set.txt')
        self.property_path = os.path.join(args.data_dir, 'attribute_set.txt')
        self.model_path = os.path.join(args.output_dir, 'model.pt')
        self.log_path = os.path.join(args.output_dir, 'train_log.log')
        self.res_path = os.path.join(args.output_dir, 'test_res.txt')
        self.dev_res_path = os.path.join(args.output_dir, 'dev_res_')

        self.entity2id, self.id2entity = mc_utils.generate_entity_property_idx(self.entity_path)
        self.label2id={'0': 0, '-1': 0, '1':1}
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        self.property2id, self.id2property = mc_utils.generate_entity_property_idx(self.property_path)
        self.train2id = mc_utils.generate_data_idx(self.train_path, self.entity2id, self.label2id)
        self.test2id = mc_utils.generate_data_idx(self.test_path, self.entity2id, self.label2id)
        self.dev2id = mc_utils.generate_data_idx(self.dev_path, self.entity2id, self.label2id)
        if self.args.embed_flag == 'tran':
            self.pre_ent_embeds, self.pre_rel_embeds = mc_utils.load_trans_embeddings(self.args.pre_ent_embeds_path,
                                                                                      self.args.pre_rel_embeds_path)

        elif self.args.embed_flag == 'ten':
            self.pre_ent_embeds, self.pre_rel_embeds = mc_utils.load_ten_embeddings(self.args.pre_ten_embeds_path,
                                                                                    self.entity2id,
                                                                                    self.property2id,
                                                                                    self.args.embed_size)
        elif self.args.embed_flag == 'concat':
            self.pre_ent_embeds_trans, self.pre_rel_embeds_trans = mc_utils.load_trans_embeddings(self.args.pre_ent_embeds_path,
                                                                                                  self.args.pre_rel_embeds_path)
            self.pre_ent_embeds_ten, self.pre_rel_embeds_ten = mc_utils.load_ten_embeddings(self.args.pre_ten_embeds_path,
                                                                                            self.entity2id,
                                                                                            self.property2id,
                                                                                            self.args.embed_size)

            entity_embedding_matrix = torch.zeros(len(self.entity2id)+1, self.args.embed_size*2)
            pro_embedding_matrix = torch.zeros(len(self.property2id)+1, self.args.embed_size*2)
            for i, embed in enumerate(self.pre_ent_embeds_ten):
                entity_embedding_matrix[i] = torch.cat((embed, self.pre_ent_embeds_trans[i]))
            for j, embed in enumerate(self.pre_rel_embeds_ten):
                pro_embedding_matrix[j] = torch.cat((embed, self.pre_rel_embeds_trans[j]))
            self.pre_ent_embeds = entity_embedding_matrix
            self.pre_rel_embeds = pro_embedding_matrix

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        
    def train(self):
        logger.info('********** Training *********')
        self.model = MLP_model(self.entity2id, self.property2id, self.pre_ent_embeds, self.pre_rel_embeds, self.args)
        # self.model = SSN_model(self.entity2id, self.property2id, self.pre_ent_embeds, self.pre_rel_embeds, self.args)
        if self.args.use_cuda:
            self.model = self.model.cuda()

        if self.args.optim_method == "Adagrad":
            self.optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                           weight_decay=self.args.weight_decay)
        elif self.args.optim_method == "Adadelta":
            self.optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                            weight_decay=self.args.weight_decay)
        elif self.args.optim_method == "Adam":
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                        weight_decay=self.args.weight_decay)
        elif self.args.optim_method == "SGD":
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr,
                                       weight_decay=self.args.weight_decay)

        res_list = []
        avg_f1_list = []
        best_model_list = []
        count = 0
        for epoch in range(self.args.epoch):
            results = 0.0
            perm = torch.randperm(len(self.train2id))
            train2id = self.train2id[perm]
            train_batches = mc_utils.get_batches(train2id, self.args.batch_size)
            for batch in train_batches:
                batch = batch.cuda() if self.args.use_cuda else batch
                self.optimizer.zero_grad()
                loss = self.model(batch)
                results += loss.data
                loss.backward()
                self.optimizer.step()
            logger.info("Loss after the %d iter : %f" % (epoch, results))
            res_list.append(results)
            if self.model_path is not None:
                logger.info('Saving and Testing on dev ...')
                self.save()
                best_model_list.append(self.model)
                avg_f1 = self.dev(epoch)
                avg_f1_list.append(avg_f1)
        best_dev_f1 = max(avg_f1_list)
        max_idx = avg_f1_list.index(best_dev_f1)
        logger.info('Best Epoch: %d, Dev loss: %f' % (max_idx, best_dev_f1))
        self.model = best_model_list[max_idx]
        self.save()
        if self.log_path is not None:
            log_file = open(self.log_path, "w", encoding='utf-8')
            for i, res in enumerate(res_list):
                log_file.write("Epoch: %d Loss: %f" % (i, res)+'\n')

    def dev(self, epoch):
        avg_f1 = 0
        if self.model_path is not None:
            self.load_model()
        if self.dev_res_path is not None:
            dev_res = open(self.dev_res_path + str(epoch) + '.txt', 'w', encoding='utf-8')
            dev2id_batches = mc_utils.get_batches(self.dev2id, self.args.batch_size)
            true_labels = []
            predict_labels = []
            test_results = 0.0
            for batch in dev2id_batches:
                batch = batch.cuda() if self.args.use_cuda else batch
                true_batch_label = batch[:, -1]
                loss, predict_batch_labels = self.model.test(batch)
                test_results += loss.data
                true_labels.append(true_batch_label)
                predict_labels.append(predict_batch_labels)

                for i, t in enumerate(batch):
                    dev_res.write('triplet: '+self.id2entity[t[0].item()]+' '+self.id2entity[t[1].item()]+'\n')
                    dev_res.write('true_label: '+str(t[-1].item())+'\n')
                    dev_res.write('predict_label: '+str(predict_batch_labels[i].item())+'\n')

            dev_res.write("Loss: "+str(test_results.item())+'\n')
            mc_results = classification_report(torch.cat(true_labels).cpu(), torch.cat(predict_labels).cpu(),
                                               target_names=['non', 'metaphor'])
            pre = precision_score(torch.cat(true_labels).cpu(), torch.cat(predict_labels).cpu(), average=None)[1]
            rec = recall_score(torch.cat(true_labels).cpu(), torch.cat(predict_labels).cpu(), average=None)[1]
            f1 = f1_score(torch.cat(true_labels).cpu(), torch.cat(predict_labels).cpu(), average=None)[1]
            logger.info("Dev P,R, F1 after the %d iter: %f,%f,%f" % (epoch, pre, rec, f1))
            dev_res.write(mc_results)
            dev_res.close()
        return f1

    def test(self):
        self.model = SSN_model(self.entity2id, self.property2id, self.pre_ent_embeds, self.pre_rel_embeds, self.args)
        logger.info("Testing ...")
        if self.model_path is not None:
            self.load_model()
        if self.res_path is not None:
            test_res = open(self.res_path, 'w', encoding='utf-8')
            test2id_batches = mc_utils.get_batches(self.test2id, self.args.batch_size)
            true_labels = []
            predict_labels = []
            test_results = 0.0
            for batch in test2id_batches:
                batch = batch.cuda() if self.args.use_cuda else batch
                true_batch_label = batch[:, -1]
                loss, predict_batch_labels = self.model.test(batch)
                test_results += loss.data
                true_labels.append(true_batch_label)
                predict_labels.append(predict_batch_labels)

                for i, t in enumerate(batch):
                    test_res.write('triplet: ' + self.id2entity[t[0].item()] + ' ' + self.id2entity[t[1].item()] + '\n')
                    test_res.write('true_label: ' + str(t[-1].item()) + '\n')
                    test_res.write('predict_label: ' + str(predict_batch_labels[i].item()) + '\n')

            test_res.write("Loss: " +str(test_results.item())+'\n')
            mc_results = classification_report(torch.cat(true_labels).cpu(), torch.cat(predict_labels).cpu(),
                                               target_names=['non', 'metaphor'])
            test_res.write(mc_results)
            test_res.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument('--pre_ent_embeds_path', default=None, type=str, )
    parser.add_argument('--pre_rel_embeds_path', default=None, type=str)
    parser.add_argument('--pre_ten_embeds_path', default=None, type=str)
    parser.add_argument('--embed_flag', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--use_cuda', default=False, type=bool)
    parser.add_argument('--epoch', default=0, type=int)
    parser.add_argument('--batch_size', default=65, type=int)
    parser.add_argument('--embed_size', default=200, type=int)
    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--target_size', default=2, type=int)
    parser.add_argument('--optim_method', default='SGD', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    args = parser.parse_args()
    
    classifier = Classify(args)
    classifier.train()

    classifier.test()


if __name__ == '__main__':
    main()
    
