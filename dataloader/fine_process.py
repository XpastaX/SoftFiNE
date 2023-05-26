import json
import torch
import config
from operator import itemgetter
from util.util import read_json
from transformers import AutoTokenizer


def text2label(data, cfg=config):  # use BIOES
    tokenizer = AutoTokenizer.from_pretrained(cfg.plm)
    tokenizer.add_tokens(['[START]', '[END]', '[O]', '[X]', '[B]', '[I]', '[E]', '[S]'])
    reconstruct = []
    for ID, item in enumerate(data):
        item['id'] = ID
        item['seq_label'] = ['[O]'] * len(item['text'])
        for category in item['label'].keys():
            for entity in item['label'][category].keys():
                for entity_idx in item['label'][category][entity]:
                    start_idx, end_idx = entity_idx
                    if start_idx == end_idx:
                        item['seq_label'][start_idx] = f'[S-{category}]'
                    else:
                        item['seq_label'][start_idx] = f'[B-{category}]'
                        item['seq_label'][end_idx] = f'[E-{category}]'
                        if end_idx - start_idx > 1:
                            for i in range(start_idx + 1, end_idx):
                                item['seq_label'][i] = f'[I-{category}]'
        item['seq_idx'] = [cfg.LabelInfo.label2id[x] for x in item['seq_label']]
        seq_label_with_se = ['[START]'] + item['seq_label'] + ['[END]']
        seq_idx_with_se = [cfg.LabelInfo.label2id['[START]']] + item['seq_idx'] + [cfg.LabelInfo.label2id['[END]']]

        split_text = [char.replace(' ', '[UNK]') for char in list(item['text'])]

        split_text_aug = [char.replace(' ', '[UNK]') for char in list(item['text'])]
        for i, token_label in enumerate(item['seq_label']):
            if token_label != '[O]':
                split_text_aug[i] = f"[{token_label.split('-')[0][1:]}]"

        reconstruct.append({'id': item['id'],  # sample ID
                            'text': item['text'],  # original sample text
                            'label': item['label'],  # original label
                            'split_text': split_text,
                            'split_text_aug': split_text_aug,
                            'seq_label': item['seq_label'],  # label, e.g. [B-Person, I-Person, E-person, O, O, O]
                            'seq_idx': item['seq_idx'],
                            'seq_label_with_se': seq_label_with_se,
                            'seq_idx_with_se': torch.tensor(seq_idx_with_se),
                            })
    return reconstruct


def collect_data(dataset_type, cfg=config):
    data = text2label(read_json(cfg.DataInfo.data_path[dataset_type]), cfg)
    torch.save(data, cfg.DataInfo.data_cache_path[dataset_type])
