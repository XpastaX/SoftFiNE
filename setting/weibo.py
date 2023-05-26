from util.util import check_dir, complete_labels
import math


class Dataset:
    dataset_name = 'Weibo'
    categories = ['PER.NAM', 'PER.NOM', 'LOC.NAM', 'LOC.NOM', 'GPE.NAM', 'ORG.NAM', 'ORG.NOM']
    num_lv, multi_label, completion, cate_lv_dict = complete_labels(categories)
    special_label = ['[SPACE]', '[START]', '[END]', '[O]', '[X]']

    data_path_root = r"data/%s/" % dataset_name
    data_path = {'train': data_path_root + 'train.json',
                 'valid': data_path_root + 'dev.json',
                 'test': data_path_root + 'test.json'}
    num_sample = {'train': 1350, 'valid': 270, 'test': 270}

    plain_model_path = 'data/plain/'
    check_dir(plain_model_path)
    cache_path_root = data_path_root + 'cache/'
    check_dir(cache_path_root)

    data_cache_path = {'train': cache_path_root + 'train.pk', 'valid': cache_path_root + 'valid.pk',
                       'test': cache_path_root + 'test.pk'}


class Exp:
    token_fea_dim = 768
    fea_loss_type = 'KLD'
    use_hi_decoder = None
    use_aug = False
    use_soft_label = True
    use_trainable_label = True
    max_length = 512
    batch_size = 4
    shuffle = True
    num_workers = 0
    prefetch_factor = 2
    persistent_workers = True

    epoch = 12
    warm_up_ratio = 0.1
    start_train = 40
    start_eval = 1
    lr = 2e-5
    decay = 1e-1
    drop_out = 0.5
    steps_per_eval = math.ceil(Dataset.num_sample['train'] / batch_size / 2)
    penalty_rate = 3
    loss2_ratio = 0.01
    loss3_ratio = 0.01


# class Exp: # F1:73.18
# -----------------PARAMETERS--------------------
#           seed:	42
#           name:	Weibo20220419003956_1
#     train_path:	data/Weibo/train.json
#     valid_path:	data/Weibo/dev.json
#     test_path:	data/Weibo/test.json
#   dataset_name:	Weibo
#  force_process:	False
#      loss_type:	KLD
#       use_soft:	True
#          batch:	4
#        max_len:	512
#          epoch:	12
#  warm_up_ratio:	0.1
#    start_train:	4
#             lr:	2e-05
#          decay:	0.1
#       drop_out:	0.5
# steps_per_eval:	169
#   penalty_rate:	3.0
#    loss2_ratio:	0.01
#    loss3_ratio:	0.01


#
#
# -----------------PARAMETERS--------------------
#           seed: 42
#           name: Weibo20220323101950
#   dataset_name: Weibo
#  force_process: False
#          batch: 4
#        max_len: 512
#          epoch: 12
#  warm_up_ratio: 0.1
#    start_train: 2
#             lr: 1e-05
#          decay: 0.1
#       drop_out: 0.5
# steps_per_eval: 169
#    loss2_ratio: 0.1
#    loss3_ratio: 0.05
#
# ********* NAME: Weibo20220323101950 BEST EVAL F1 is 0.7289002557544757 **********
# ********* EVAL RESULTS E8 S169  **********
#  acc: 0.7234 - recall: 0.7345 - cut_acc: 0.7386 - cut_recall: 0.7500 - pred_acc: 0.9794 - cut_f1: 0.7442 - f1: 0.7289
# ************* ENTITY RESULTS  *************
#                  ---GPE.NAM results---
#  acc: 0.6000 - recall: 0.9231 - f1: 0.7273
#                  ---LOC.NAM results---
#  acc: 0.6250 - recall: 0.8333 - f1: 0.7143
#                  ---LOC.NOM results---
#  acc: 0.6667 - recall: 0.3333 - f1: 0.4444
#                  ---ORG.NAM results---
#  acc: 0.4211 - recall: 0.3404 - f1: 0.3765
#                  ---ORG.NOM results---
#  acc: 0.7500 - recall: 0.6000 - f1: 0.6667
#                  ---PER.NAM results---
#  acc: 0.7692 - recall: 0.7778 - f1: 0.7735
#                  ---PER.NOM results---
#  acc: 0.7857 - recall: 0.7933 - f1: 0.7895
# *******************************************
#
# model loaded from output/Weibo20220323101950/ckpt/Weibo_best_plain.ckpt
# ********* NAME: Weibo20220323101950 BEST TEST F1 is 0.7194066749072929 **********
# ********* EVAL RESULTS E0 S0  **********
#  acc: 0.7405 - recall: 0.6995 - cut_acc: 0.7735 - cut_recall: 0.7308 - pred_acc: 0.9572 - cut_f1: 0.7515 - f1: 0.7194
# ************* ENTITY RESULTS  *************
#                  ---GPE.NAM results---
#  acc: 0.7288 - recall: 0.9149 - f1: 0.8113
#                  ---LOC.NAM results---
#  acc: 0.7273 - recall: 0.4211 - f1: 0.5333
#                  ---LOC.NOM results---
#  acc: 1.0000 - recall: 0.3333 - f1: 0.5000
#                  ---ORG.NAM results---
#  acc: 0.6800 - recall: 0.4359 - f1: 0.5312
#                  ---ORG.NOM results---
#  acc: 0.8571 - recall: 0.3529 - f1: 0.5000
#                  ---PER.NAM results---
#  acc: 0.7838 - recall: 0.7699 - f1: 0.7768
#                  ---PER.NOM results---
#  acc: 0.7175 - recall: 0.7384 - f1: 0.7278
# *******************************************

