import os
import time
import random
import json
import torch
import numpy as np


class Args:
    num_features = 250
    rand_seed = 42
<<<<<<< HEAD
    group = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH', 'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']
=======

>>>>>>> vim_branch
    def __init__(self):

        self.root_dir = '/home/pekew/code/EEGNet/data'
        self.train_fold = 'all'
        self.subjects_type = 'inter'  # intra | inter 表示验证方法是被试内还是被试间
        self.valid_method = 'kfold'  # 是否使用k折验证
        self.search = False  # 是否搜索网络参数
        self.cpu = not torch.cuda.is_available()
<<<<<<< HEAD
        self.early_stop = 20
        self.band = 30 # 频带数
        self.num_nodes = self.band
        self.num_epochs_gnn = 20
        self.num_epochs_video = 16
        self.l1_reg = 0.001
        self.l2_reg = 0.001
        self.lr = 0.001
        self.dropout = 0.5
        self.num_hiddens = 50
        self.num_layers = 2
        self.n_vids = 24
=======
        self.model_save_path = '/home/pekew/code/EEGNet/results/models'

>>>>>>> vim_branch
        self.num_classes = 2
        self.num_workers = 8
        self.batch_size = 16

        # gnn参数 - 总
        self.band = 30  # 频带数
        self.num_nodes = self.band
        self.num_epochs = 25
        self.l1_reg = 0.0001
        self.l2_reg = 0.0005
        self.lr = 0.0003
        self.dropout = 0.3
        self.num_hiddens = 64
        self.num_layers = 3
        self.clip_norm = 3
        self.node_learnable = True

        # 下面这两个参数一般需要一起修改
        self.mod = ['eeg']  # 数据集加载的模态: 可选项: eeg | optical | original | motion
        self.model_mod = 'eeg_group'  # cnn | eeg_group
        self.n_folds = None
        self.n_per = None
        self.sec = None
        self.data_root_dir = None
        self.now_time = None
        self.model_path = None
        self.sub_list = None
<<<<<<< HEAD
        self.node_learnable = True
        self.eeg_hidden_size = 32
        self.eeg_dropout = 0.5
        self.data_sampler_strategy = 'down'
        self.optimizer = 'Adam'
        # CNN 参数
        self.channels1 = 16
        self.channels2 = 32
        
        # VAE 参数
        self.hidden_size = 256
        self.vae_dropout = 0.5
        
        # CNNVAE 参数
        # todo edge是对称矩阵 减少参数
        self.edge_hidden_size = 900
        self.node_hidden_size = 250*30
        
        
        self.search = True # 是否搜索网络参数


    def init_range_gnn(self):
        # node_learnable
        self.node_learnable_list = [True, False]
        # eeg_hidden_size
        self.eeg_hidden_size_list = [16, 32, 64, 128]
        # eeg_dropout
        self.eeg_dropout_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        # num_layers
        self.num_layers_list = [1, 2, 4, 6]
        # batch_size
        self.batch_size_list = [16, 32, 64, 128]
        # data_strategy
        self.data_sampler_strategy_list = ['down', 'up']
        # optimizer
        self.optimizer_list = ['Adam', 'SGD']
        # group: 随机取group的50% - 100%
        self.group_list = [
            random.sample(self.group, random.randint(len(self.group) // 2, len(self.group)))
            for _ in range(5)
        ]
        # rand_seed
        self.random_seed_list = [random.randint(0, 100) for _ in range(10)]
        # epoch
        self.num_epochs_gnn = [5, 10, 15, 20]
        # l1_reg
        self.l1_reg_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.05]
        # l2_reg
        self.l2_reg_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.05]
        # lr
        self.lr_list = [0.0001, 0.001, 0.01, 0.1]
        import itertools
        combinations = itertools.product(
            self.node_learnable_list,
            self.eeg_hidden_size_list,
            self.eeg_dropout_list,
            self.num_layers_list,
            self.batch_size_list,
            self.data_sampler_strategy_list,
            self.optimizer_list,
            self.group_list,
            self.random_seed_list,
            self.num_epochs_gnn,
            self.l1_reg_list,
            self.l2_reg_list,
            self.lr_list
        )
        param_dict = []
        for combo in combinations:
            param_dict.append({
                'node_learnable': combo[0],
                'eeg_hidden_size': combo[1],
                'eeg_dropout': combo[2],
                'num_layers': combo[3],
                'batch_size': combo[4],
                'data_sampler_strategy': combo[5],
                'optimizer': combo[6],
                'group': combo[7],
                'rand_seed': combo[8],
                'num_epochs_gnn': combo[9],
                'l1_reg': combo[10],
                'l2_reg': combo[11],
                'lr': combo[12]
            })
        return param_dict
=======
>>>>>>> vim_branch

        self.channels1 = 16
        self.channels2 = 32

        # VAE 参数
        self.hidden_size = 256
        self.vae_dropout = 0.5
        self.bae_latent_dim = 90

        # CNNVAE 参数
        # todo edge是对称矩阵 减少参数
        self.edge_hidden_size = 900
        self.node_hidden_size = 250*30

        self.num_epochs_video = 50

        # self.video_include = ['original']
        #
        # 对比学习参数
        self.temperature = 0.05

        # cross modal multi attention
        self.num_heads = 8
        self.atn_hidden_dim = 256

        # loss
        self.alpha = 1
        self.beta = 1
        self.gamma = 1
        self.delta = 1

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
