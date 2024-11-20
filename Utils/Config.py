import os
import time
import random
import json
import torch
import numpy as np

class Args:
    num_features = 250
    rand_seed = 42
    def __init__(self):
        self.root_dir = '/home/pekew/code/EEGNet/data'
        self.train_fold = 'all'
        self.subjects_type = 'inter' # intra | inter 表示验证方法是被试内还是被试间
        self.valid_method = 'kfold' # 是否使用k折验证
        self.search = False # 是否搜索网络参数
        self.cpu = not torch.cuda.is_available()
        
        self.num_classes = 2
        self.num_workers = 8
        self.batch_size = 16
        
        # gnn参数 - 总
        self.band = 30 # 频带数
        self.num_nodes = self.band
        self.num_epochs = 25
        self.l1_reg = 0.0001
        self.l2_reg = 0.0005
        self.lr = 0.0003
        self.dropout = 0.3
        self.num_hiddens = 64
        self.num_layers = 3
        self.clip_norm = 3
        
        self.mod = ['eeg'] # 数据集加载的模态: 可选项: eeg | optical | original | motion
        self.group_mod = 'random' # 正则化图的分组方法: 可选项: gender | random
        self.n_folds = None
        self.n_per = None
        self.sec = None
        self.data_root_dir = None
        self.now_time = None
        self.model_path = None
        self.sub_list = None


    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)


def init():
    config = Args()
    torch.manual_seed(config.rand_seed)
    torch.cuda.manual_seed(config.rand_seed)
    torch.cuda.manual_seed_all(config.rand_seed)
    np.random.seed(config.rand_seed)
    random.seed(config.rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.synchronize()
    torch.backends.cudnn.enabled = False
    return config