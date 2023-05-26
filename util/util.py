import os
import random
import torch
import numpy as np
import time
import json


def read_json(path):
    data = []
    with open(path, 'r', encoding='UTF-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def complete_labels(categories):
    num_lv = 1
    for item in categories:
        if len(item.split('/')[1:]) > num_lv:
            num_lv = len(item.split('/')[1:])
    cate_lv_dict = {i + 1: {} for i in range(num_lv)}

    for item in categories:
        cates  = item.split('/')[1:]
        for lv in range(num_lv):
            if len(cates) > lv:
                tmp = '/' + '/'.join(cates[:lv+1])
                if tmp not in cate_lv_dict[lv+1]:
                    cate_lv_dict[lv+1][tmp] = 0
                cate_lv_dict[lv+1][tmp]+=1

    multi_label = []
    for lv in cate_lv_dict:
        multi_label+=list(cate_lv_dict[lv])
    completion = []
    for item in multi_label:
        if item not in categories:
            completion.append(item)
    return num_lv,multi_label,completion,cate_lv_dict

def generate_labels(classes,completion=None):
    tag_list = ['[PAD]', '[START]', '[END]', '[O]', '[X]']
    completion_list = []
    id2tag = {}
    tag2id = {}

    for cls in classes:
        tag_list.append('[B-%s]' % cls)
        tag_list.append('[I-%s]' % cls)
        tag_list.append('[E-%s]' % cls)
        tag_list.append('[S-%s]' % cls)
    for i, item in enumerate(tag_list):
        id2tag[i] = item
        tag2id[item] = i
    if completion is None:
        return tag_list, id2tag, tag2id
    else:
        for cls in completion:
            completion_list.append('[B-%s]' % cls)
            completion_list.append('[I-%s]' % cls)
            completion_list.append('[E-%s]' % cls)
            completion_list.append('[S-%s]' % cls)
        completion_list += ['[B]', '[I]', '[E]', '[S]']
        start = len(id2tag)
        for i, item in enumerate(completion_list):
            id2tag[i+start] = item
            tag2id[item] = i+start
        return tag_list, id2tag, tag2id, completion_list

def check_dir(path, creat=True):
    if not os.path.exists(path):
        if creat:
            os.makedirs(path)
            print('Folder %s has been created.' % path)
            return True
        else:
            return False
    else:
        return True


def set_seed(seed):
    """
    :param seed:
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def write_log(txt, path, prt=True):
    if prt:
        print(txt)
    with open(path, 'a') as file:
        file.writelines(txt)


class ProgressBar(object):
    '''
    custom progress bar
    '''

    def __init__(self, n_total, width=30, desc='Training'):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'

        show_bar += time_info
        if len(info) != 0:
            show_info = f'{show_bar} ' + \
                        "-".join([f' {key}: {value:.4f} ' for key, value in info.items()])
            print(show_info, end='')
        else:
            print(show_bar, end='')
