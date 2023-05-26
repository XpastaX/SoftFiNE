import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import config
from model.decoder.classifier import BaseClassifier
from operator import itemgetter
import datetime


class Plain(nn.Module):
    def __init__(self, cfg=config):
        super(Plain, self).__init__()
        self.name = cfg.name
        self.cfg = cfg
        self.O = cfg.LabelInfo.label2id['[O]']
        self.para_device = None
        self.fea_size = cfg.Exp.token_fea_dim
        drop_out = cfg.Exp.drop_out
        self.out_size = cfg.LabelInfo.num_labels
        self.id2target = cfg.LabelInfo.id2label
        # extractor
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.plm)
        self.tokenizer.add_tokens(['[START]', '[END]', '[O]', '[X]', '[B]', '[I]', '[E]', '[S]'])
        self.encoder = AutoModel.from_pretrained(cfg.plm)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        # decoder, output dim is number of label types plus CLS and SEP
        self.decoder = BaseClassifier(self.fea_size, [], self.out_size, drop_out, self.name)
        self.cos = nn.CosineSimilarity()
        self.fea_loss = None
        self.init_loss(self.cfg.Exp.fea_loss_type)
        # loss type
        if self.cfg.Exp.use_soft_label:
            self.criterion = self.softCE
            self.use_soft = True
        else:
            self.use_soft = False
            self.criterion = torch.nn.CrossEntropyLoss()

    def init_loss(self, loss_type):
        if loss_type == 'KLD':
            self.fea_loss = self.KLD
        elif loss_type == 'cosine':
            self.fea_loss = self.cosine
        else:
            print('The loss type is not implemented')
            raise NotImplementedError

    def cosine(self, fea_input, target):
        return 1 - self.cos(fea_input, target.detach()).sum() / len(fea_input)

    def KLD(self, fea_input, target):
        return F.kl_div(F.log_softmax(fea_input, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')

    def softCE(self, ce_input, target):
        logprobs = F.log_softmax(ce_input, dim=-1)
        return -(target * logprobs).sum() / ce_input.shape[0]
        # return -(target/target.sum(-1)[:,None] * logprobs * target.sum(-1)[:,None]).sum() / ce_input.shape[0]
        # return -(F.softmax(target, dim=-1) * logprobs).sum() / ce_input.shape[0]

    def forward(self, text, label, aug=None, soft_label=None):
        if self.use_soft:
            ground_truth = soft_label
        else:
            ground_truth = label
        if self.para_device is None:
            self.para_device = next(self.parameters()).device

        ground_truth = ground_truth.to(self.para_device)
        token = {key: text[key].to(self.para_device) for key in text.keys()}
        mask_index = token['attention_mask'].contiguous().reshape(-1).eq(1)
        fea = self.encoder(**token)['last_hidden_state']
        x = self.decoder(fea).contiguous().view(-1, self.out_size)[mask_index]  # batch,seq,labels
        loss1 = self.criterion(x, ground_truth)
        loss2 = torch.tensor(0, device=self.para_device)
        loss3 = torch.tensor(0, device=self.para_device)

        if aug is not None:  # then use augmentation data
            token_aug = {key: aug[key].to(self.para_device) for key in aug.keys()}
            output_aug = self.encoder(**token_aug)
            fea_aug = output_aug['last_hidden_state']
            x_aug = self.decoder(fea_aug)
            x_aug = x_aug.view(-1, self.out_size)[mask_index]
            loss2 = self.criterion(x_aug, ground_truth)
            active = label.gt(self.O)
            loss3 = self.fea_loss(fea_aug.view(-1, self.fea_size)[mask_index][active],
                                  fea.view(-1, self.fea_size)[mask_index][active])

        return loss1 + self.cfg.Exp.loss2_ratio * loss2 + self.cfg.Exp.loss3_ratio * loss3, loss1, loss2, loss3

    def predict(self, text, tokenized=False):
        if self.para_device is None:
            self.para_device = next(self.parameters()).device
        token = {key: text[key].to(self.para_device) for key in text.keys()}
        mask = token["attention_mask"]
        output = self.encoder(**token)  # batch,seq,768
        x = output['last_hidden_state']
        x = self.decoder(x)  # batch,seq,labels
        pred = torch.argmax(x, dim=2)
        pred_list = [sample[mask[i] == 1][1:-1] for i, sample in enumerate(pred)]
        # sequence = [list(itemgetter(*sample.int().tolist())(self.id2target)) for i, sample in enumerate(pred_list)]

        return pred_list

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        for param in self.encoder.parameters():
            param.requires_grad = train_module[0]
        for param in self.decoder.parameters():
            param.requires_grad = train_module[1]

    def save(self, save_name, print_path=False, save_mode=False):
        # save all modules
        if save_mode:
            encoder_ckpt_path = self.cfg.dir_ckpt + save_name + '.encoder'
            torch.save(self.encoder.state_dict(), encoder_ckpt_path)
            if print_path:
                print('encoder saved at:')
                print(encoder_ckpt_path)

            decoder_ckpt_path = self.cfg.dir_ckpt + save_name + '.decoder'
            torch.save(self.decoder.state_dict(), decoder_ckpt_path)
            if print_path:
                print('decoder saved at:')
                print(decoder_ckpt_path)

        ckpt_path = self.cfg.dir_ckpt + save_name + '.ckpt'
        torch.save(self.state_dict(), ckpt_path)

        if print_path:
            print('model saved at:')
            print(ckpt_path)

    def load(self, load_name, direct=False, load_mode=False):
        if load_mode:
            if direct:
                encoder_ckpt_path = load_name[0]
                decoder_ckpt_path = load_name[1]
            else:
                encoder_ckpt_path = self.cfg.dir_ckpt + load_name + '.encoder'
                decoder_ckpt_path = self.cfg.dir_ckpt + load_name + '.decoder'
            self.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=next(self.parameters()).device))
            self.decoder.load_state_dict(torch.load(decoder_ckpt_path, map_location=next(self.parameters()).device))
            print(f'encoder loaded from {encoder_ckpt_path}')
            print(f'decoder loaded from {decoder_ckpt_path}')
        else:
            if direct:
                ckpt_path = load_name
            else:
                ckpt_path = self.cfg.dir_ckpt + load_name + '.ckpt'

            self.load_state_dict(torch.load(ckpt_path, map_location=next(self.parameters()).device))
            print(f'model loaded from {ckpt_path}')
