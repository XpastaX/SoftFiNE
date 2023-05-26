# This is to test the accuracy of the segmentation
import args as arguments
import config
from util.util import set_seed
from trainer.plain_trainer import trainer as T_pre


def run_pre_test(args, cfg=config):
    cfg = arguments.init_config(args, cfg)
    set_seed(cfg.seed)
    trainer = T_pre(cfg=cfg, use_aug=cfg.Exp.use_aug)
    trainer.train()


if __name__ == '__main__':
    run_pre_test(arguments.parser.parse_args())
