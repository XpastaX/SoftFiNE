from util.util import check_dir, complete_labels
import math

categories = ['NAME', 'CONT', 'RACE', 'TITLE', 'EDU', 'ORG', 'PRO', 'LOC']


class DataInfo:
    dataset_name = 'Resume'

    num_lv, multi_label, completion, cate_lv_dict = complete_labels(categories)
    special_label = ['[SPACE]', '[START]', '[END]', '[O]', '[X]']

    data_path_root = r"data/%s/" % dataset_name
    data_path = {'train': data_path_root + 'train.json',
                 'valid': data_path_root + 'dev.json',  # no dev set available
                 'test': data_path_root + 'test.json'}
    num_sample = {'train': 3821, 'valid': 463, 'test': 477}

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
    use_trainable_label = False
    max_length = 512
    batch_size = 8
    shuffle = True
    num_workers = 0
    prefetch_factor = 2
    persistent_workers = True

    epoch = 12
    warm_up_ratio = 0.1
    start_train = 1
    start_eval = 1
    lr = 2e-5
    decay = 1e-2
    drop_out = 0.5
    steps_per_eval = math.ceil(DataInfo.num_sample['train'] / batch_size / 4)
    penalty_rate = 4
    loss2_ratio = 0.01
    loss3_ratio = 0.005
