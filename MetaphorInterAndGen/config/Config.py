import utils
import torch.optim as optim
import sys
import random
import torch
import logging

stdout_back = sys.stdout
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Config(object):
    def __init__(self):
        self.trainPath = None
        self.testPath = None
        self.devPath = None
        self.conceptAttrPath = None
        self.entityPath = None
        self.propertyPath = None
        self.sample_ent_candi_path = None
        self.sample_attr_candi_path = None
        self.test_entity_candi_path = None
        self.test_attr_candi_path = None
        self.tycWordsPath = None
        self.embeddingPath = None
        self.modelPath = None
        self.logPath = None
        self.devLogPath = None
        self.mi_resPath = None
        self.mg_resPath = None
        self.dev_mi_resPath = None
        self.dev_mg_resPath = None
        self.preTrainedModelPath = None
        self.use_cuda = False
        self.ent_size = 200
        self.rel_size = 200
        self.batch_size = 64
        self.max_attr_size = 50
        self.matrix_rand_init = True
        self.p_norm = 1
        self.epochs = 1000
        self.margin = 1.0
        self.opt_method = "SGD"
        self.optimizer = None
        self.lr = 0.001
        self.lr_decay = 0.000
        self.weight_decay = 0.000

        self.alpha = 0.0
        self.beta = 0.0

        self.regul_rate = 0.0
        self.norm_flag = False

    def _init(self):
        logger.info("Initializing ...")
        self.entity2id, self.id2entity, self.entid2tags = utils.generate_entity_property_idx(self.entityPath)
        self.property2id, self.id2property, self.proid2tags = utils.generate_entity_property_idx(self.propertyPath)
        self.entid2tycid = utils.generate_entity_tyc_idx(self.tycWordsPath, self.entity2id)
        self.train2id = utils.generate_data_idx(self.trainPath, self.entity2id, self.property2id)
        self.train2id_set = set([' '.join(map(str, t)) for t in self.train2id])  # use for sampling
        self.conid2attrid = utils.generate_conceptid_to_attributesid(self.conceptAttrPath, self.entity2id,
                                                                          self.property2id, self.max_attr_size)
        self.conAttr2id, self.conAttr2id_set = utils.generate_concept_attributes_idx(self.conceptAttrPath,
                                                                                          self.entity2id,
                                                                                          self.property2id)
        self.dev2id = utils.generate_data_idx(self.devPath, self.entity2id, self.property2id)
        self.test2id = utils.generate_data_idx(self.testPath, self.entity2id, self.property2id)

        self.test_entity_candidate_ids = utils.read_sample_candidates(self.test_entity_candi_path, self.entity2id)
        self.test_attr_candidate_ids = utils.read_sample_candidates(self.test_attr_candi_path, self.property2id)

        self.sample_ent_cand_ids = utils.read_sample_candidates(self.sample_ent_candi_path, self.entity2id)
        self.sample_attr_cand_ids = utils.read_sample_candidates(self.sample_attr_candi_path, self.property2id)

        self.trainTotal = len(self.train2id)
        self.conceptAttrTotal = len(self.conid2attrid)
        self.devTotal = len(self.dev2id)
        self.testTotal = len(self.test2id)
        self.entityTotal = len(self.entity2id)
        self.propertyTotal = len(self.property2id)

        # tencent init
        if self.embeddingPath is not None:
            self.ent_embeddings = utils.load_embeddings(self.entity2id, self.embeddingPath,
                                                             self.entityTotal,
                                                             self.ent_size)
            self.rel_embeddings = utils.load_embeddings(self.property2id, self.embeddingPath,
                                                             self.propertyTotal,
                                                             self.rel_size)

        self.dev2id_batches = utils.get_batches(self.dev2id, self.batch_size)
        self.test2id_batches = utils.get_batches(self.test2id, self.batch_size)

    def set_cuda(self, cuda):
        self.use_cuda = cuda

    def set_epochs(self, epoches):
        self.epochs = epoches

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_ent_size(self, size):
        self.ent_size = size

    def set_rel_size(self, size):
        self.rel_size = size

    def set_max_attr_size(self, size):
        self.max_attr_size = size

    def set_p_norm(self, norm):
        self.p_norm = norm

    def set_matrix_randn_init(self, flag):
        self.matrix_rand_init = flag

    def set_lr(self, lr):
        self.lr = lr

    def set_margin(self, margin):
        self.margin = margin

    def set_mk_weight(self, alpha):
        self.alpha = alpha

    def set_ca_weight(self, beta):
        self.beta = beta

    def set_regul_rate(self, rate):
        self.regul_rate = rate

    def set_norm_flag(self, flag):
        self.norm_flag = flag

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_concept_attribute_path(self, path):
        self.conceptAttrPath = path

    def set_train_path(self, path):
        self.trainPath = path

    def set_dev_path(self, path):
        self.devPath = path

    def set_test_path(self, path):
        self.testPath = path

    def set_entity_path(self, path):
        self.entityPath = path

    def set_property_path(self, path):
        self.propertyPath = path

    def set_sample_ent_candi_path(self, path):
        self.sample_ent_candi_path = path

    def set_sample_attr_candi_path(self, path):
        self.sample_attr_candi_path = path

    def set_test_ent_candi_path(self, path):
        self.test_entity_candi_path = path

    def set_test_attr_candi_path(self, path):
        self.test_attr_candi_path = path

    def set_tyc_words_path(self, path):
        self.tycWordsPath = path

    def set_embed_path(self, path):
        self.embeddingPath = path

    def set_model_path(self, path):
        self.modelPath = path

    def set_log_path(self, path):
        self.logPath = path

    def set_devlog_path(self, path):
        self.devLogPath = path

    def set_mi_resPath(self, path):
        self.mi_resPath = path

    def set_mg_resPath(self, path):
        self.mg_resPath = path

    def set_dev_mi_resPath(self, path):
        self.dev_mi_resPath = path

    def set_dev_mg_resPath(self, path):
        self.dev_mg_resPath = path

    def set_pre_trained_model_path(self, path):
        self.preTrainedModelPath = path

    def save(self, epoch):
        torch.save(self.trainModel.state_dict(), self.modelPath + str(epoch) + '.pt')

    def load_model(self, epoch):
        self.trainModel.load_state_dict(torch.load(self.modelPath + str(epoch) + '.pt'))
        # self.trainModel.load_state_dict(torch.load(self.modelPath+str(epoch)+'.pt', map_location=lambda
        # storage, loc: storage)) # to cpu

    def load_pretrained_model(self):
        self.trainModel.load_state_dict(torch.load(self.preTrainedModelPath))

    def set_model(self, model):
        logger.info("Setting Model ...")
        self.trainModel = model(config=self)
        if self.use_cuda:
            self.trainModel = self.trainModel.cuda()

        if self.optimizer is not None:
            pass
        elif self.opt_method == "Adagrad":
            self.optimizer = optim.Adagrad(self.trainModel.parameters(), lr=self.lr, lr_decay=self.lr_decay,
                                           weight_decay=self.weight_decay)
        elif self.opt_method == "Adadelta":
            self.optimizer = optim.Adadelta(self.trainModel.parameters(), lr=self.lr)
        elif self.opt_method == "Adam":
            self.optimizer = optim.Adam(self.trainModel.parameters(), lr=self.lr)
        else:
            self.optimizer = optim.SGD(self.trainModel.parameters(), lr=self.lr)

    def run(self):
        logger.info("Strat Training ...")
        res_list = []
        dev_res_list = []
        log_file = open(self.logPath, "w")
        log_file.close()
        dev_log_file = open(self.devLogPath, 'w')
        dev_log_file.close()
        for epoch in range(self.epochs):
            results = 0.0
            random.shuffle(self.train2id)
            self.train2id_batches = utils.get_batches(self.train2id, self.batch_size)
            self.conAttr2id_random = utils.conAttr_choice(self.conAttr2id)
            self.conAttr2id_batches = utils.get_batches(self.conAttr2id_random, self.batch_size)
            for (i, ca_batch) in enumerate(self.conAttr2id_batches):
                self.optimizer.zero_grad()
                self.STbatch = utils.get_tuples(ca_batch, self.sample_ent_cand_ids, self.sample_attr_cand_ids,
                                                     self.conid2attrid,
                                                     self.conAttr2id_set, self.train2id_set, True)
                loss_ca = self.trainModel(ca_flag=True)
                if i < len(self.train2id_batches):
                    batch = self.train2id_batches[i]
                    self.STbatch = utils.get_tuples(batch, self.sample_ent_cand_ids, self.sample_attr_cand_ids,
                                                         self.conid2attrid,
                                                         self.conAttr2id_set, self.train2id_set)
                    loss_mk = self.trainModel()
                    loss = self.alpha * loss_mk + self.beta * loss_ca
                else:
                    loss = self.beta * loss_ca
                results = results + loss.data / self.batch_size
                loss.backward()
                self.optimizer.step()

            logger.info("Loss after the %d iter : %f" % (epoch, results))
            res_list.append(results)
            with open(self.logPath, "a", encoding="utf-8") as log_file:
                log_file.write("Epoch: " + str(epoch) + " Loss: " + str(results.item()) + '\n')
            if epoch % 2 == 0:
                logger.info("Saving and Testing on Dev !")
                if self.modelPath is not None:
                    self.save(epoch)
                dev_res = 0.0
                for batch in self.dev2id_batches:
                    self.STbatch = utils.get_tuples(batch, self.sample_ent_cand_ids, self.sample_attr_cand_ids,
                                                         self.conid2attrid,
                                                         self.conAttr2id_set, self.train2id_set)
                    dev_loss = self.trainModel()
                    dev_res = dev_res + dev_loss.data / self.batch_size
                logger.info("dev_loss after the %d iter : %f" % (epoch, dev_res))
                dev_res_list.append(dev_res)
                with open(self.devLogPath, 'a', encoding='utf-8') as dev_log_file:
                    dev_log_file.write("Epoch: " + str(epoch) + " Loss: " + str(dev_res.item()) + '\n')
        self.save(self.epochs)
        # test
        best_epoch = dev_res_list.index(min(dev_res_list)) * 2
        logger.info("Best Epoch: %d" % (best_epoch))
        self.test(self.test2id, self.mi_resPath, self.mg_resPath, best_epoch)

    def test(self, test2id, mi_resPath, mg_resPath, epoch):
        logger.info("Testing ! ")
        if self.modelPath is not None:
            self.load_model(epoch)
        if mi_resPath is not None:
            pre_r_res_file = open(mi_resPath + str(epoch) + '.txt', "w", encoding="utf-8")
            sys.stdout = pre_r_res_file
            utils.link_prediction_with_cilin_tag(test2id, self.trainModel, self.id2entity,
                                                      self.id2property,
                                                      self.test_entity_candidate_ids, self.test_attr_candidate_ids,
                                                      self.entid2tags, self.proid2tags, 'p', self.use_cuda)

            sys.stdout = stdout_back

        if mg_resPath is not None:
            pre_t_res_file = open(mg_resPath + str(epoch) + '.txt', "w", encoding="utf-8")
            sys.stdout = pre_t_res_file
            utils.link_prediction_with_cilin_tag(test2id, self.trainModel, self.id2entity,
                                                      self.id2property,
                                                      self.test_entity_candidate_ids, self.test_attr_candidate_ids,
                                                      self.entid2tags, self.proid2tags, 't', self.use_cuda)
            sys.stdout = stdout_back
