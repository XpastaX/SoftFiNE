import config
from collections import Counter


def get_entity_bioes(seq, id2label):
    """Gets entities from sequence.
    note: BIOES
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    current_length = 0
    for indx, tag in enumerate(seq):
        if not isinstance(tag, str):
            tag = id2label[int(tag)]
        if tag.startswith("[") and tag not in ['[START]', '[END]']:
            tag = tag.strip('[]')
        # check single char entity
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = [-1, -1, -1]
            continue
        # check Beginning
        if tag.startswith("B-") and current_length == 0:
            # start to count
            current_length += 1
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        # check Intermediate
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                current_length += 1
            else:
                current_length = 0

        elif tag.startswith('E-') and chunk[1] != -1 and current_length > 0:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                current_length = 0
                chunk[2] = indx
                chunks.append(chunk)
                chunk = [-1, -1, -1]
        elif tag == 'O':
            current_length = 0
            chunk = [-1, -1, -1]
            continue
        else:
            current_length = 0
            chunk = [-1, -1, -1]
            # print('Unknown Tag Found in BIOES!')
    return chunks


def get_entities(seq, id2label, markup='BIO'):
    '''
    :param seq:
    :param id2label:
    :param markup:
    :return:
    '''
    assert markup in ['BIOES']
    return get_entity_bioes(seq, id2label)


class SeqEntityScore(object):
    def __init__(self, id2tag, markup='BIOES', label='plain'):
        self.origins = []
        self.founds = []
        self.rights = []
        self.right_cut = []
        self.ori_cut = []
        self.id2label = id2tag
        self.markup = markup
        self.reset()
        self.label = label

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []
        self.right_cut = []
        self.ori_cut = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])

        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        right_cut = len(self.right_cut)
        recall, precision, f1 = self.compute(origin, found, right)
        cut_recall = 0 if origin == 0 else right_cut / origin
        cut_acc = 0 if found == 0 else right_cut / found
        cut_f1 = 0. if cut_recall + cut_acc == 0 else (2 * cut_acc * cut_recall) / (cut_acc + cut_recall)
        pred_acc = 0 if right_cut == 0 else right / right_cut
        return {'acc': precision, 'recall': recall, 'cut_acc': cut_acc, 'cut_recall': cut_recall, 'pred_acc': pred_acc,
                'cut_f1': cut_f1, 'f1': f1, }, class_info

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = get_entities(label_path, self.id2label, self.markup)
            pre_entities = get_entities(pre_path, self.id2label, self.markup)
            ori_cut = [l_e[1:] for l_e in label_entities]
            self.origins.extend(label_entities)
            self.ori_cut.extend(ori_cut)
            self.founds.extend(pre_entities)
            self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
            self.right_cut.extend([pre_entity for pre_entity in pre_entities if pre_entity[1:] in ori_cut])
