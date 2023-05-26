import argparse
import config
import torch
from util.util import generate_labels
from util.util import check_dir
from util.util import write_log
from util.label_process import process
import random

# parser used to read argument
parser = argparse.ArgumentParser(description='xxxxx')

# ==================================
#            Universal
# ==================================
parser.add_argument('--seed', type=int, default=config.seed)
parser.add_argument('--device', type=str, default=config.device, help='device to run on')
parser.add_argument('--label_type', type=str, default=config.LabelInfo.label_type, help='BIO,BIOS or BIOES')
parser.add_argument('--data', type=str, default='Resume', help='dataset name, e.g., Weibo')
parser.add_argument('--force_process', action="store_true", help='force preprocess')
# ==================================
#      Model & Training  Settings
# ==================================
parser.add_argument('--loss_type', type=str, default='not_defined', help='loss type for feas')
parser.add_argument('--plm', type=str, default='', help='plm model to use')
parser.add_argument('--batch_size', type=int, default=0, help='batch size')
parser.add_argument('--max_length', type=int, default=0, help='max_length')
parser.add_argument('--epoch', type=int, default=0, help='num of epoch')
parser.add_argument('--start_train', type=int, default=0, help='start_train')
parser.add_argument('--start_eval', type=int, default=0, help='start_eval')
parser.add_argument('--lr', type=float, default=0, help='initial learning rate')
parser.add_argument('--decay', type=float, default=0, help='decay factor')
parser.add_argument('--drop_out', type=float, default=-1, help='decay factor')
parser.add_argument('--steps_per_eval', type=int, default=0, help='initial learning rate')
parser.add_argument('--soft', action="store_true", help='use soft label')
parser.add_argument('--aug', action="store_true", help='use aug')
parser.add_argument('--hi', type=str, default='undefined', help='one_hot, soft')
parser.add_argument('--penalty', type=float, default=0, help='penalty factor')
parser.add_argument('--loss2', type=float, default=0, help='loss2 ratio')
parser.add_argument('--loss3', type=float, default=0, help='loss3 ratio')


# ==================================
#      Initialization
# ==================================
def init_config(args, cfg):
    # update universal parameters
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.label_type = args.label_type
    cfg.name = args.data + cfg.ID
    cfg.dataset = args.data
    # whether force processing data
    if args.force_process:
        cfg.force_process = True
    # update pretrained language model to use
    if args.plm != '':
        cfg.plm = cfg.dir_plm + args.plm + '/'
    # check ID to avoid duplication
    addon = 0
    while check_dir(cfg.dir_output + cfg.name + '_' + str(addon) + r'/', creat=False):
        addon += 1
    cfg.name += '_' + str(addon)
    # update output path
    cfg.dir_output += cfg.name + r'/'
    cfg.dir_ckpt = cfg.dir_output + r'ckpt/'
    cfg.file_log = cfg.dir_output + r'log.txt'
    cfg.file_config = cfg.dir_output + r'config.torch'
    check_dir(cfg.dir_ckpt)
    # overwrite Dataset and Exp with corresponding data settings
    cfg = overwrite_config(cfg)
    # update exp settings in args
    cfg = update_Exp(cfg, args)
    # update label info
    cfg = update_LabelInfo(cfg)
    # save and print config
    save_config(cfg)
    return cfg


def overwrite_config(cfg):
    if cfg.dataset == "FiNE":
        import setting.fine as target
    elif cfg.dataset == "Weibo":
        import setting.weibo as target
    elif cfg.dataset == "ONTONOTES":
        import setting.ontonotes as target
    elif cfg.dataset == "Resume":
        import setting.resume as target
    else:
        raise NotImplementedError(f'Data {cfg.dataset} is not Implemented')
    cfg.DataInfo = target.DataInfo
    cfg.Exp = target.Exp
    cfg.LabelInfo.categories = target.categories
    return cfg


def update_Exp(cfg, args):
    cfg.Exp.fea_loss_type = args.loss_type if args.loss_type != 'not_defined' else cfg.Exp.fea_loss_type
    cfg.Exp.batch_size = args.batch_size if args.batch_size != 0 else cfg.Exp.batch_size
    cfg.Exp.max_length = args.max_length if args.max_length != 0 else cfg.Exp.max_length
    cfg.Exp.epoch = args.epoch if args.epoch != 0 else cfg.Exp.epoch
    cfg.Exp.start_train = args.start_train if args.start_train != 0 else cfg.Exp.start_train
    cfg.Exp.start_eval = args.start_eval if args.start_eval != 0 else cfg.Exp.start_eval
    cfg.Exp.lr = args.lr if args.lr != 0 else cfg.Exp.lr
    cfg.Exp.decay = args.decay if args.decay != 0 else cfg.Exp.decay
    cfg.Exp.drop_out = args.drop_out if args.drop_out != -1 else cfg.Exp.drop_out
    cfg.Exp.steps_per_eval = args.steps_per_eval if args.steps_per_eval != 0 else cfg.Exp.steps_per_eval
    cfg.Exp.penalty_rate = args.penalty if args.penalty != 0 else cfg.Exp.penalty_rate
    cfg.Exp.loss2_ratio = args.loss2 if args.loss2 != 0 else cfg.Exp.loss2_ratio
    cfg.Exp.loss3_ratio = args.loss3 if args.loss3 != 0 else cfg.Exp.loss3_ratio
    if args.soft:
        cfg.Exp.use_soft_label = True
    if args.aug:
        cfg.Exp.use_aug = True
    return cfg


def update_LabelInfo(cfg):
    cfg = process(cfg)
    return cfg


def save_config(cfg):
    log = '-----------------PARAMETERS--------------------\n'
    log += f'          seed:\t{cfg.seed}\n' \
           f'          name:\t{cfg.name}\n' \
           f"    train_path:\t{cfg.DataInfo.data_path['train']}\n" \
           f"    valid_path:\t{cfg.DataInfo.data_path['valid']}\n" \
           f"    test_path:\t{cfg.DataInfo.data_path['test']}\n" \
           f'  dataset_name:\t{cfg.dataset}\n' \
           f' force_process:\t{cfg.force_process}\n' \
           f'     loss_type:\t{cfg.Exp.fea_loss_type}\n' \
           f'      use_soft:\t{cfg.Exp.use_soft_label}\n' \
           f'       use_aug:\t{cfg.Exp.use_aug}\n' \
           f'         batch:\t{cfg.Exp.batch_size}\n' \
           f'       max_len:\t{cfg.Exp.max_length}\n' \
           f'         epoch:\t{cfg.Exp.epoch}\n' \
           f' warm_up_ratio:\t{cfg.Exp.warm_up_ratio}\n' \
           f'   start_train:\t{cfg.Exp.start_train}\n' \
           f'            lr:\t{cfg.Exp.lr}\n' \
           f'         decay:\t{cfg.Exp.decay}\n' \
           f'      drop_out:\t{cfg.Exp.drop_out}\n' \
           f'steps_per_eval:\t{cfg.Exp.steps_per_eval}\n' \
           f'  penalty_rate:\t{cfg.Exp.penalty_rate}\n' \
           f'   loss2_ratio:\t{cfg.Exp.loss2_ratio}\n' \
           f'   loss3_ratio:\t{cfg.Exp.loss3_ratio}\n'
    write_log(log, cfg.file_log)
