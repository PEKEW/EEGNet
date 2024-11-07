import os
import time
import random
import json
import torch
import numpy as np

class Args:
    def __init__(self):
        self.root_dir = '/home/pekew/code/EEGNet/data'
        self.rand_seed = 42
        self.train_fold = 'all'
        self.subjects_type = 'inter' # intra | inter 表示验证方法是被试内还是被试间
        self.valid_method = 'kfold' # 是否使用k折验证
        self.cpu = not torch.cuda.is_available()
        self.early_stop = 20
        self.band = 30 # 频带数
        self.num_nodes = self.band
        self.num_epochs = 100
        self.l1_reg = 0.001
        self.l2_reg = 0.001
        self.lr = 0.001
        self.dropout = 0.5
        self.num_hiddens = 900
        self.num_layers = 2
        self.n_vids = 24
        self.num_classes = 2
        self.n_subs = 15
        self.num_workers = 8
        self.num_features = 250
        self.batch_size = 32
        self.clip_norm = 20
        self.mod = ['eeg'] # 数据集加载的模态: 可选项: eeg | optical | original | motion
        self.group_mod = 'gender' # 正则化图的分组方法: 可选项: gender | random
        self.n_folds = None
        self.n_per = None
        self.sec = None
        self.data_root_dir = None
        self.now_time = None
        self.model_path = None
        self.sub_list = None
        self.search = False # 是否搜索网络参数


    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def init():
    config = Args()
    torch.manual_seed(config.rand_seed)
    torch.cuda.manual_seed(config.rand_seed)
    torch.cuda.manual_seed_all(config.rand_seed)
    np.random.seed(config.rand_seed)
    random.seed(config.rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    config.n_folds = 3 if config.valid_method == 'kfold' else config.n_subs
    config.n_per = 5
    config.sec = 30
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config.data_root_dir = os.path.join(current_dir, f'./Data/{config.band}bands/smooth_{config.n_vids}')
    config.now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    config.model_path = os.path.join(current_dir, f'./result/_{config.now_time}_{config.subjects_type}_{config.n_vids}')
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    json.dump(config.to_dict(), open(f'{config.model_path}/args_{config.now_time}.json', 'w'))
    return config