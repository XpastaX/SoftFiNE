import config as default_cfg
import torch.nn.functional as F
import torch


def reform_label(tag_list, has_slash, lv):
    seg = tag_list[0]
    cate = tag_list[1:lv + 1]
    if has_slash:
        cate = '/' + '/'.join(cate)
    else:
        cate = '-'.join(cate)
    return '-'.join([seg, cate]).strip('-/')


class label_generator(object):
    def __init__(self, cfg=default_cfg):
        self.cfg = cfg
        self.l2r, self.l2p = self.init_soft_label_dict(cfg.LabelInfo.labels)

    def init_soft_label_dict(self, label_list):
        label_list = [item.strip('[]') for item in label_list]
        label2rel_soft = {}
        label2preference = {}
        for idx, label in enumerate(label_list):
            # first generate one hot
            label2rel_soft[label] = [0.] * len(label_list)
            label2rel_soft[label][idx] = 1.
            label2preference[label] = torch.zeros((self.cfg.LabelInfo.num_labels, self.cfg.LabelInfo.num_labels))
        # adding rel information
        for key in label2rel_soft:
            for idx, label in enumerate(label_list):
                if key == label: continue
                l = split_tags(label)
                k = split_tags(key)
                min_len = min(len(l), len(k))
                # check correct
                is_related = False
                for j, item in enumerate(k):
                    if j >= len(l): break
                    if item == l[j]:
                        is_related = True
                if not is_related: continue
                # check miss
                miss = max(len(k) - len(l), 0)
                # check wrong
                wrong = 0
                for item in l:
                    if item not in k:
                        wrong += 1
                penalty = self.cfg.Exp.penalty_rate ** (miss + 2 * wrong)
                label2rel_soft[key][idx] = 1 / penalty
        for key in label2rel_soft:
            soft_label = torch.tensor(label2rel_soft[key])
            tmp = soft_label.expand((len(soft_label),len(soft_label)))
            label2preference[key] = tmp.T-tmp
        return label2rel_soft, label2preference

    def rel_label(self, seq):
        return [self.l2r[label.strip('[]')] for label in seq]

    def preference_matrix(self, seq):
        return [self.l2p[label.strip('[]')] for label in seq]


def split_tags(tag):
    if '/' in tag:
        seg, cate = tag.split('-')
        cate = cate.split('/')
        hierarchical_tag = [seg] + cate[1:]
    else:
        hierarchical_tag = tag.split('-')
    return hierarchical_tag
