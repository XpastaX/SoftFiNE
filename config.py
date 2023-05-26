import torch
import datetime
from util.util import check_dir

# Universal
seed = 42
ID = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
dataset = 'Default'
name = dataset + ID  # will be used to create separate ckpt dir to save parameters and logs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
force_process = False

# path
dir_output = 'output/'
dir_ckpt = dir_output + r'ckpt/'
file_log = dir_output + r'log.txt'
file_config = dir_output + r'config.torch'
dir_plm = r"data/plm/"
check_dir(dir_plm)
plm = dir_plm + 'bert-base-chinese/'


class LabelInfo:
    label_type = 'BIOES'
    special_label = ['[PAD]', '[START]', '[END]', '[O]', '[X]']  # for classification
    special_token = special_label + [f'[{tag}]' for tag in label_type.replace('O', '')]  # for augmentation
    label_lv0 = special_label + [f'[{tag}]' for tag in label_type.replace('O', '')]

    # ======================================
    # will be updated during parsing args
    # ======================================
    categories = []
    labels = []
    multi_labels = {0: [i.strip('[]') for i in label_lv0]}

    # statics
    num_categories = 0
    num_labels = 0
    num_multi_labels = 0
    num_lv_categories = 0
    num_lv_labels = num_lv_categories + 1
    lv_shape = ()
    # conversion between index and labels
    id2label = {}
    label2id = {}
    label2mul = {}
    label2mul_flat = {}
    hi_retrieve_mul = {}
    hi_mul_mask = torch.tensor([])
    # label mask is used to retrieve original classes from a multilabel matrix.
    # It should be a bool of tensor A[label_mask] = labels
    label_mask = []
    reorder = torch.tensor([])

# ======================================
#       Dataset and Exp Settings
# everything below will be overwritten
#   by specific setting in /setting/...
#   during parsing args
# ======================================

class DataInfo:
    data_path_root = ''
    data_path = {}
    plain_model_path = ''
    num_sample = {'train': 0, 'valid': 0, 'test': 0}
    cache_path_root = data_path_root + 'cache/'
    data_cache_path = {'train': '', 'valid': '', 'test': ''}


class Exp:
    use_multi_label = False
    use_hi_decoder = None
    use_aug = False
    use_soft_label = False
    use_trainable_label = False

    token_fea_dim = 768
    fea_loss_type = 'xxx'
    sample_loss_type = 'xxx'
    max_length = 0
    batch_size = 2
    shuffle = True
    num_workers = 4
    prefetch_factor = None
    persistent_workers = True

    epoch = 0
    start_eval = 1
    warm_up_ratio = 0
    start_train = 0
    lr = 0
    decay = 0
    drop_out = 0

    steps_per_eval = 0
    penalty_rate = 0
    loss2_ratio = 0
    loss3_ratio = 0
    hi_loss_ratio = 0
