import itertools
import random
import torch
import numpy as np


class Args:
    num_features = 250
    rand_seed = 42
    group = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH',
            'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']

    # TODO: important: declear all hyperparameters with different part
    def __init__(self):

        self.root_dir = '/home/pekew/code/EEGNet/data'
        self.train_fold = 'all'
        self.subjects_type = 'inter'
        self.valid_method = 'kfold'
        self.search = False
        self.cpu = not torch.cuda.is_available()
        self.early_stop = 20
        self.band = 30
        self.num_nodes = self.band
        self.gnn_num_epochs = 20
        self.cnn_num_epochs = 16
        self.mcdis_num_epochs = 20
        self.l1_reg = 0.001
        self.l2_reg = 0.001
        self.lr = 0.001
        self.dropout = 0.5
        self.num_layers = 2
        self.n_vids = 24
        self.model_save_path = '/home/pekew/code/EEGNet/results/models'
        self.num_classes = 2
        self.num_workers = 8
        self.batch_size = 16
        self.band = 30
        self.num_nodes = self.band
        self.num_epochs = 25
        self.l1_reg = 0.0001
        self.l2_reg = 0.0005
        self.lr = 0.0003
        self.gnn_hiddens_size = 64
        self.gnn_num_layers = 3
        self.clip_norm = 3
        self.node_learnable = True
        # self.mod = ['eeg']  #  eeg | optical | original | motion
        self.model_mod = 'all'  # cnn | eeg_group | all
        self.n_folds = None
        self.n_per = None
        self.sec = None
        self.data_root_dir = None
        self.now_time = None
        self.model_path = None
        self.sub_list = None
        self.node_learnable = True
        self.gnn_hidden_size = 32
        self.gnn_dropout = 0.5
        self.data_sampler_strategy = 'up'
        self.optimizer = 'Adam'
        self.channels1 = 16
        self.channels2 = 32
        self.channels3 = 64
        self.embed_dim = 32
        self.bae_dropout = 0.5
        # TODO: improve edge is symmetric matrix reduce parameters
        self.edge_hidden_size = 900
        self.node_hidden_size = 250*30
        self.search = True
        self.epoch_diff = 100
        self.diff_steps = 100
        self.channels0 = 16
        self.channels1 = 32
        self.bae_hidden_size = 512
        self.bae_latent_size = 64
        self.bae_latent_dim = 90
        self.cnn_num_epochs = 50
        self.temperature = 0.05
        self.num_heads = 8
        self.atn_hidden_dim = 256
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.delta = 1
        self.batch_size_all = 16
        self.clip_norm_all = 5

        self.nce_edge_lr = 1e-3
        self.nce_node_lr = 1e-3
        self.nce_gnn_lr = 1e-3
        
        self.dtf_path='/home/pekew/code/EEGNet/results/models/dtf.pth'
        self.bae_att_dropout = 0.5
        
    def init_range_gnn(self):
        self.node_learnable_list = [True, False]
        self.eeg_hidden_size_list = [16, 32, 64, 128]
        self.eeg_dropout_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.num_layers_list = [1, 2, 4, 6]
        self.batch_size_list = [16, 32, 64, 128]
        self.data_sampler_strategy_list = ['down', 'up']
        self.optimizer_list = ['Adam', 'SGD']
        # group: 50% - 100%
        self.group_list = [
            random.sample(self.group, random.randint(
                len(self.group) // 2, len(self.group)))
            for _ in range(5)
        ]
        self.random_seed_list = [random.randint(0, 100) for _ in range(10)]
        self.gnn_num_epochs = [5, 10, 15, 20]
        self.l1_reg_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.05]
        self.l2_reg_list = [0.001, 0.005, 0.01, 0.05, 0.1, 0.05]
        self.lr_list = [0.0001, 0.001, 0.01, 0.1]
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
            self.gnn_num_epochs,
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
                'gnn_num_epochs': combo[9],
                'l1_reg': combo[10],
                'l2_reg': combo[11],
                'lr': combo[12]
            })
        return param_dict

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
