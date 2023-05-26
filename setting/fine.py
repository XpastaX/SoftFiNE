from util.util import check_dir, complete_labels
import math

categories = [
    # 生物
    '/biology/flora_fauna', '/biology/pet',
    # 事件
    '/event/historical_event',
    '/event/social_event/entertainment_event', '/event/social_event/exhibition',
    '/event/social_event/social_activity', '/event/social_event/sports_activities',
    # 食物
    '/food/dishes', '/food/material',
    # 地名
    '/location/site', '/location/site/city', '/location/site/scenic',
    # 医疗
    '/medicine/cure', '/medicine/drug', '/medicine/illness',
    # 机构名
    '/organization/company', '/organization/company/3c_brand', '/organization/company/consumer_goods_brand',
    '/organization/company/vehicle_brand', '/organization/social_institution/association',
    '/organization/social_institution/financing_institution',
    '/organization/social_institution/government_sector',
    '/organization/social_institution/public_institution', '/organization/team/e_sport_team',
    '/organization/team/entertainment_team', '/organization/team/sport_team',
    # 人名
    '/person',
    '/person/e_sport', '/person/economy', '/person/entertainment', '/person/history', '/person/politics',
    '/person/sport', '/person/virtual',
    # 商品
    '/product/consumables/daily_necessaries',
    '/product/consumables/finance_product', '/product/consumables/technological_product',
    '/product/tools/vehicle', '/product/tools/vehicle_parts', '/product/tools/weapon',
    '/product/virtual_good',
    # 时间
    '/time/common_time', '/time/dynasty', '/time/festival',
    # 作品
    '/55work/artistic_creation/annals', '/work/artistic_creation/arts',
    '/work/artistic_creation/literature', '/work/game_software/game', '/work/game_software/software',
    '/work/video/comic', '/work/video/movie', '/work/video/show', '/work/video/teleplay',
    '/work/video/tv_show']


class DataInfo:
    data_path_root = r"data/FiNE/"
    data_path = {'train': data_path_root + 'new_train.json',
                 'valid': data_path_root + 'dev.json',
                 'test': data_path_root + 'test.json'}
    num_sample = {'train': 23000, 'valid': 3000, 'test': 4000}
    plain_model_path = 'data/plain/'
    check_dir(plain_model_path)
    cache_path_root = data_path_root + 'cache/'
    check_dir(cache_path_root)
    data_cache_path = {'train': cache_path_root + 'train.pk', 'valid': cache_path_root + 'valid.pk',
                       'test': cache_path_root + 'test.pk'}


class Exp:
    token_fea_dim = 768
    fea_loss_type = 'KLD'
    sample_loss_type = 'BCE'
    use_hi_decoder = 'one_hot'
    use_aug = False
    use_soft_label = True
    batch_size = 16
    max_length = 512
    shuffle = True
    num_workers = 0
    prefetch_factor = 2
    persistent_workers = True

    epoch = 20
    warm_up_ratio = 0.05
    start_train = 5
    start_eval = 5
    lr = 2e-5
    decay = 1e-2
    drop_out = 0.5
    steps_per_eval = math.ceil(DataInfo.num_sample['train'] / batch_size / 4)
    penalty_rate = 2
    loss2_ratio = 0.1
    loss3_ratio = 0.1
    hi_loss_ratio = [0.1, 0.1, 0.1, 1]
