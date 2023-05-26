import torch
from torch.utils.data import Dataset, DataLoader
import config
from dataloader.fine_process import collect_data
from transformers import AutoTokenizer
import os
from dataloader.gen_soft_label import label_generator


class FiNEdataset(Dataset):
    def __init__(self, dataset_type, cfg=config):
        self.label_list = None
        self.label_dict = None
        self.data_path = cfg.DataInfo.data_cache_path[dataset_type]
        self.data = None
        self.size = cfg.DataInfo.num_sample[dataset_type]
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.plm)
        self.tokenizer.add_tokens(['[START]', '[END]', '[O]', '[X]', '[B]', '[I]', '[E]', '[S]'])
        self.cfg = cfg
        self.max_len = cfg.Exp.max_length if dataset_type == 'train' else 512
        self.get_label = label_generator(cfg=cfg)

    def __len__(self):
        return self.size

    def load_data(self):
        self.data = torch.load(self.data_path)
        self.label_dict = {}
        for index, sample in enumerate(self.data):
            for label in sample['label']:
                if '/' in label:
                    lv1 = label.split('/')[1]
                else:
                    lv1 = label
                if lv1 not in self.label_dict:
                    self.label_dict[lv1] = [index]
                else:
                    self.label_dict[lv1].append(index)
        self.label_list = list(self.label_dict.keys())

    def collate_fcn(self, batch):
        id_list = [self.data[idx]['id'] for idx in batch]
        label_idx_list = [tuple(self.data[idx]['seq_idx'][0:self.max_len - 2]) for idx in batch]

        # fix the labels for over length seq
        label_idx_list_se = []
        rel_label_se = []
        for idx in batch:
            tmp_label_idx_list_se = self.data[idx]['seq_idx_with_se'][0:self.max_len].copy()
            tmp_label_idx_list_se[-1] = self.data[idx]['seq_idx_with_se'][-1]
            label_idx_list_se.append(torch.tensor(tmp_label_idx_list_se))
            tmp_label_list_se = self.data[idx]['seq_label_with_se'][0:self.max_len].copy()
            tmp_label_list_se[-1] = self.data[idx]['seq_label_with_se'][-1]
            rel_label_se.append(torch.tensor(self.get_label.rel_label(tmp_label_list_se)))

        # prepare labels
        processed_rel_label = []
        processed_label = []
        for idx in batch:
            item = self.data[idx]
            processed_rel_label += self.get_label.rel_label(item['seq_label_with_se'][0:self.max_len])
            # processed_rel_preference += self.get_label.preference_matrix(item['seq_label_with_se'][0:self.max_len])
            processed_label += item['seq_idx_with_se'][0:self.max_len]

        # prepare multilabels
        multi_label_flat = [self.data[idx]['multi_label_with_se_flat'] for idx in batch]
        multi_label_idx = [self.data[idx]['multi_label_with_se'] for idx in batch]
        # prepare tokenized texts
        text_split_list = [self.data[idx]['split_text'] for idx in batch]
        text_split_list_aug = [self.data[idx]['split_text_aug'] for idx in batch]
        tokenized = self.tokenizer(text_split_list, padding=True, truncation=True, max_length=self.max_len,
                                   is_split_into_words=True, return_tensors="pt")
        tokenized_aug = self.tokenizer(text_split_list_aug, padding=True, truncation=True, max_length=self.max_len,
                                       is_split_into_words=True, return_tensors="pt")

        processed_multi_label_flat = []
        processed_multi_label_idx = []
        for item in multi_label_flat:
            processed_multi_label_flat += item
        for item in multi_label_idx:
            processed_multi_label_idx += item
        samples = {
            'label_idx_list': label_idx_list,
            'processed_multi_label_flat': torch.tensor(processed_multi_label_flat).float(),
            'processed_multi_label_idx': torch.tensor(processed_multi_label_idx).long(),
            'processed_rel_label': torch.tensor(processed_rel_label),
            'processed_rel_preference': 0,
            'processed_label': torch.tensor(processed_label),
            'tokenized': tokenized,
            'tokenized_label_entity': tokenized_aug
        }

        return samples

    def __getitem__(self, idx):
        if self.data is None:
            self.load_data()
        return idx


def prepro(cfg=config):
    if cfg.force_process:
        process = ['train', 'valid', 'test']
        cfg.force_process=False
    else:
        process = []
        for key in ['train', 'valid', 'test']:
            if not os.path.isfile(cfg.DataInfo.data_cache_path[key]):
                process.append(key)
    for dataset_type in process:
        print('processing data %s' % dataset_type)
        collect_data(dataset_type, cfg)


def get_dataloader(dataset_type, cfg=config):
    prepro(cfg)  # preprocess data
    dataset = FiNEdataset(dataset_type, cfg)
    if cfg.Exp.num_workers <= 0:
        return DataLoader(dataset, batch_size=cfg.Exp.batch_size,
                          shuffle=cfg.Exp.shuffle if dataset_type == 'train' else False,
                          sampler=None,
                          batch_sampler=None, num_workers=cfg.Exp.num_workers, collate_fn=dataset.collate_fcn,
                          pin_memory=True, drop_last=False, timeout=0,
                          worker_init_fn=None)
    else:
        return DataLoader(dataset, batch_size=cfg.Exp.batch_size,
                          shuffle=cfg.Exp.shuffle if dataset_type == 'train' else False,
                          sampler=None,
                          batch_sampler=None, num_workers=cfg.Exp.num_workers, collate_fn=dataset.collate_fcn,
                          pin_memory=True, drop_last=False, timeout=0,
                          worker_init_fn=None, prefetch_factor=cfg.Exp.prefetch_factor,
                          persistent_workers=True)
