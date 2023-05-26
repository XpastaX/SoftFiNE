import config
import torch


def process(cfg):
    cfg = generate_labels(cfg)
    cfg = generate_multi_labels(cfg)
    cfg.LabelInfo.num_labels = len(cfg.LabelInfo.labels)
    cfg.LabelInfo.num_categories = len(cfg.LabelInfo.categories)
    cfg = generate_label_mask(cfg)
    return cfg


def get_categories(cfg=config):
    return cfg.LabelInfo.categories


def generate_labels(cfg=config):
    label_list = cfg.LabelInfo.special_label.copy()
    categories = get_categories(cfg)
    for cls in categories:
        label_list.append('[B-%s]' % cls)
        label_list.append('[I-%s]' % cls)
        label_list.append('[E-%s]' % cls)
        label_list.append('[S-%s]' % cls)
    cfg.LabelInfo.labels = label_list
    # generate indexes
    id2label = {}
    label2id = {}
    for index, label in enumerate(cfg.LabelInfo.labels):
        id2label[index] = label
        label2id[label] = index
    cfg.LabelInfo.id2label = id2label
    cfg.LabelInfo.label2id = label2id
    return cfg


def generate_multi_labels(cfg=config):
    max_lv = 1
    # if the categories are hierarchical
    if '/' in cfg.LabelInfo.categories[0]:
        for cate in cfg.LabelInfo.categories:
            lv = cate.count('/')
            if max_lv < lv:
                max_lv = lv
        lv_list = [{f'EMPTY_LV{i + 1}': 0} for i in range(max_lv)]  # add empty label for all cate levels.
        for cate in cfg.LabelInfo.categories:
            d_cate = decompose_label(cate)
            assert len(d_cate) <= max_lv
            for idx, tag in enumerate(d_cate):
                lv_list[idx][tag] = 0
        for i in range(1, max_lv + 1):
            cfg.LabelInfo.multi_labels[i] = sorted(list(lv_list[i - 1].keys()))
    else:
        cfg.LabelInfo.multi_labels[1] = get_categories(cfg) + ['EMPTY_LV1']
    cfg.LabelInfo.num_lv_categories = max_lv
    cfg.LabelInfo.num_lv_labels = max_lv + 1  # includes B,I,O,E,S
    lv_shape = [0] * cfg.LabelInfo.num_lv_labels
    for i, lv in enumerate(cfg.LabelInfo.multi_labels):
        lv_shape[i] = len(cfg.LabelInfo.multi_labels[lv])
    cfg.LabelInfo.lv_shape = lv_shape
    cfg.LabelInfo.num_multi_labels = sum(lv_shape)
    return cfg


def decompose_label(label, pad=0):
    label = label.strip('[]')
    tag = label.replace('/', '-').replace('--', '-').strip('/-')
    if pad == 0:
        return tag.split('-')
    else:
        tags = tag.split('-')
        size = len(tags)
        if size < pad:
            for i in range(pad - size):
                tags.append(f'EMPTY_LV{i + size}')
        return tags


def generate_label_mask(cfg):
    # first generate all possible labels, then calculate a mask for true labels
    dimension = cfg.LabelInfo.lv_shape
    # mask = torch.zeros(dimension).bool()
    label2mul = {}
    label2mul_flat = {}
    indexes_list = []
    hi_retrieve_mul = {}  # record the idx mapping between hi_label and multilabels
    hi_mul_mask = torch.zeros((cfg.LabelInfo.num_labels, cfg.LabelInfo.num_lv_labels)).long()
    for idx, label in enumerate(cfg.LabelInfo.labels):
        tags = decompose_label(label, pad=cfg.LabelInfo.num_lv_labels)
        indexes = [0] * cfg.LabelInfo.num_lv_labels
        idx_mapping = [0] * cfg.LabelInfo.num_lv_labels
        _counter = 0
        for lv, tag in enumerate(tags):
            indexes[lv] = cfg.LabelInfo.multi_labels[lv].index(tag)
            idx_mapping[lv] = indexes[lv] + _counter
            _counter += dimension[lv]
        # mask[tuple(indexes)] = True
        indexes_list.append(tuple(indexes))
        label2mul[label] = tuple(indexes)
        mul_label_flat = [0] * sum(cfg.LabelInfo.lv_shape)
        for i, item in enumerate(indexes):
            mul_label_flat[item + sum(cfg.LabelInfo.lv_shape[:i])] = 1
        label2mul_flat[label] = mul_label_flat
        # given a hi_label, now generate idx tensor to retrieve probabilities of the corresponding multi labels
        hi_retrieve_mul[label] = torch.tensor(idx_mapping).long()
        hi_mul_mask[idx, :] = hi_retrieve_mul[label]

    # sorted_nums = sorted(enumerate(indexes_list), key=lambda x: x[1])
    # sort_idx = [i[0] for i in sorted_nums]
    #
    # sorted_nums = sorted(enumerate(sort_idx), key=lambda x: x[1])
    # reorder = [i[0] for i in sorted_nums]
    # cfg.LabelInfo.reorder = torch.tensor(reorder).long()
    cfg.LabelInfo.label2mul_flat = label2mul_flat
    # cfg.LabelInfo.label_mask = mask
    cfg.LabelInfo.hi_mul_mask = hi_mul_mask
    cfg.LabelInfo.label2mul = label2mul  # mapping class to multi labels
    cfg.LabelInfo.hi_retrieve_mul = hi_retrieve_mul
    return cfg
