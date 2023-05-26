import torch
import config
from util.util import write_log
from util.metric import SeqEntityScore
from model.fast import Plain
# from model.tri_token import tri_token
from util.util import ProgressBar
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import logging
from dataloader.fine_loader import get_dataloader
import datetime
logging.set_verbosity_error()


class trainer(object):
    def __init__(self, use_aug=False, cfg=config):

        self.train_loader = get_dataloader('train', cfg=cfg)
        self.valid_loader = get_dataloader('valid', cfg=cfg)
        self.test_loader = get_dataloader('test', cfg=cfg)

        self.use_aug = use_aug
        self.processed_soft = 'processed_rel_label'
        id2target = cfg.LabelInfo.id2label
        self.label_key = 'label_idx_list'
        self.processed_key = 'processed_label'
        self.score = SeqEntityScore(id2target, markup=cfg.LabelInfo.label_type)
        self.model = Plain(cfg=cfg).to(cfg.device)
        self.cfg = cfg
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.cfg.Exp.decay, },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        total_step = self.cfg.Exp.epoch * len(self.train_loader)
        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.Exp.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                         num_warmup_steps=int(self.cfg.Exp.warm_up_ratio * total_step),
                                                         num_training_steps=total_step)

    def train(self):
        best_result = None
        best_f1 = 0
        pbar = ProgressBar(n_total=len(self.train_loader), desc='Training')
        self.model.train()
        print('Start Training')
        self.model.set_train([True, True])
        for epoch in range(1, self.cfg.Exp.epoch + 1):
            for step, batch in enumerate(self.train_loader, start=1):
                self.optimizer.zero_grad()

                if not self.use_aug or epoch < self.cfg.Exp.start_train:
                    loss, l1, l2, l3 = self.model(batch['tokenized'], batch[self.processed_key],
                                                  soft_label=batch[self.processed_soft])
                    pbar(step, {'epoch': epoch, 'loss': loss.item()})
                else:
                    loss, l1, l2, l3 = self.model(batch['tokenized'], batch[self.processed_key],
                                                  batch['tokenized_label_entity'],
                                                  soft_label=batch[self.processed_soft])
                    pbar(step, {'epoch': epoch, 'loss': loss.item(),
                                'l1': l1.item(), 'l2': l2.item(), 'l3': l3.item(), })

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                if epoch >= self.cfg.Exp.start_eval:
                    if step % self.cfg.Exp.steps_per_eval == 0 or step == len(self.train_loader):
                        f1_score, log = self.eval(epoch, step)
                        if best_f1 < f1_score:
                            best_f1 = f1_score
                            best_result = log
                            prt_txt = f"\nFound best f1 socre at {epoch}-{step}:{best_f1}\n"
                            self.model.save(self.cfg.dataset + '_best',
                                            save_mode=False, print_path=True)
                        else:
                            prt_txt = f"\nf1 socre at {epoch}-{step}:{f1_score}\n"
                        write_log(prt_txt, self.cfg.file_log)
        best_result = "********* NAME: %s BEST EVAL F1 is %s **********\n" % (self.cfg.name, best_f1,) + best_result
        write_log(best_result, self.cfg.file_log)

        # test result
        f1_score, log = self.test()
        log = "********* NAME: %s BEST TEST F1 is %s **********\n" % (self.cfg.name, f1_score,) + log
        write_log(log, self.cfg.file_log)

    def eval(self, epoch=0, step=0, load_best=False):
        if load_best:
            self.model.load('BEST_F1.ckpt')
        self.model.eval()
        self.score.reset()
        with torch.no_grad():
            for e_step, e_batch in enumerate(self.valid_loader):
                pred_list = self.model.predict(e_batch['tokenized'], tokenized=True)
                self.score.update(e_batch[self.label_key], pred_list)

            eval_info, entity_info = self.score.result()

            f1_score, log = self.print(eval_info, entity_info, epoch, step)

        self.model.train()
        return f1_score, log,

    def print(self, eval_info, entity_info, epoch=0, step=0):
        results = {f'{key}': value for key, value in eval_info.items()}
        log = f"********* EVAL RESULTS E{epoch} S{step}  **********\n"
        log += "-".join([f' {key}: {value:.4f} ' for key, value in results.items()])
        log += "\n************* ENTITY RESULTS  *************\n"
        for key in sorted(entity_info.keys()):
            log += "\t\t ---%s results--- \n" % key
            log += "-".join([f' {key}: {value:.4f} ' for key, value in entity_info[key].items()])
            log += "\n"
        log += '*******************************************\n'
        # print(log)
        return results['f1'], log

    def test(self, current=False):
        if not current:
            self.model.load(self.cfg.dataset + '_best')
        self.model.eval()
        self.score.reset()
        with torch.no_grad():
            for e_step, e_batch in enumerate(self.test_loader):
                pred_list = self.model.predict(e_batch['tokenized'], tokenized=True)
                self.score.update(e_batch[self.label_key], pred_list)

            eval_info, entity_info = self.score.result()
            f1_score, log = self.print(eval_info, entity_info)

        self.model.train()
        return f1_score, log,
