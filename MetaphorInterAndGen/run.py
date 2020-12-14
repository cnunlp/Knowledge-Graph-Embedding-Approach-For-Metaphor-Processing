import numpy as np
import os
import config.Config as Config
# from models.TransEMetaJoint import TransEMetaJoint
from models.TransHMetaJoint import TransHMetaJoint
# from models.TransDMetaJoint import TransDMetaJoint
# from models.RotatEMetaJoint import RotatEMetaJoint


data_dir = './data'
res_dir = './res'
# setting
con = Config.Config()
con.set_cuda(True)
con.set_train_path(os.path.join(data_dir, 'train.txt'))
con.set_concept_attribute_path(os.path.join(data_dir, 'concept_attributes.txt'))
con.set_dev_path(os.path.join(data_dir, 'dev.txt'))
con.set_test_path(os.path.join(data_dir, 'test.txt'))
con.set_entity_path(os.path.join(data_dir, 'entity_set.txt'))
con.set_property_path(os.path.join(data_dir, 'attributes_set.txt'))
con.set_tyc_words_path(os.path.join(data_dir, 'entity_tyc.txt'))
con.set_sample_ent_candi_path(os.path.join(data_dir, 'entity_set.txt'))
con.set_sample_attr_candi_path(os.path.join(data_dir, 'attributes_set.txt'))
con.set_test_ent_candi_path(os.path.join(data_dir, 'entity_set.txt'))
con.set_test_attr_candi_path(os.path.join(data_dir, 'attributes_set.txt'))
con.set_epochs(1000)
con.set_opt_method("Adadelta")
con.set_ent_size(200)
con.set_rel_size(200)
con.set_max_attr_size(500)
con.set_p_norm(1)
con.set_margin(4)
con.set_regul_rate(0.5)
con.set_norm_flag(True)
con.set_mk_weight(1.0)
con.set_ca_weight(1.0)
con.set_model_path(os.path.join(res_dir, 'models_'))
con.set_log_path(os.path.join(res_dir, 'trainlog.log'))
con.set_devlog_path(os.path.join(res_dir, 'devlog.log'))

batch = [256, 512, 1024]
lrs = np.arange(0.1, 1.0, 0.1)
for bt in batch:
    for lr in lrs:
        print(bt, lr)
        con.set_batch_size(bt)
        con.set_lr(lr)
        con.set_mi_resPath(os.path.join(res_dir, "predict_attribute_"+str(bt)+'_'+str(lr)+'_'))
        con.set_mg_resPath(os.path.join(res_dir, "predict_source_"+str(bt)+'_'+str(lr)+'_'))
        # load model
        con._init()
        con.set_model(TransHMetaJoint)
        # train and test
        con.run()
