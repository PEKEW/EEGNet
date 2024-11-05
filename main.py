import os
import copy
import time
import random
import numpy as np
import json
import torch
import Utils
from models.Utils import Results, benchmark
import sys
from Utils.test import test_working_directory


class Args:
    def __init__(self):
        self.root_dir = '/home/pekew/code/EEGNet/data'
        self.rand_seed = 42
        self.train_flod = 'all'
        self.subjects_type = 'inter'
        self.valid_method = 'kfold'
        self.auto_device_count = 5
        self.device_list = [0]
        self.device_index = -1
        self.cpu = False
        self.early_stop = 20
        self.band = 30
        self.num_nodes = 32
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

        self.n_flods = None
        self.n_per = None
        self.sec = None
        self.data_root_dir = None
        self.now_time = None
        self.model_path = None
        self.batch_size = 32

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        setattr(self, key, value)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def init():
    config = Args()

    # 设置随机种子
    torch.manual_seed(config.rand_seed)
    torch.cuda.manual_seed(config.rand_seed)
    torch.cuda.manual_seed_all(config.rand_seed)
    np.random.seed(config.rand_seed)
    random.seed(config.rand_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 计算相关参数
    config.n_flods = 3 if config.valid_method == 'kfold' else config.n_subs
    # config.n_per = round(config.n_subs / config.n_flods)
    config.n_per = 5
    config.sec = 30

    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config.data_root_dir = os.path.join(current_dir, f'./Data/{config.band}bands/smooth_{config.n_vids}')
    config.now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    config.model_path = os.path.join(current_dir, f'./result/_{config.now_time}_{config.subjects_type}_{config.n_vids}')

    # 创建模型保存目录
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)

    # 保存配置
    json.dump(config.to_dict(), open(f'{config.model_path}/args_{config.now_time}.json', 'w'))

    return config




def main(args):
    return benchmark(args)

if __name__ == '__main__':
    args = init()
    if args.train_flod == 'all':
        flod_list = np.arange(0, args.n_flods)
    else:
        flod_list = [int(args.train_flod)]
    result = Results(args)
    buc = []
    # gm = GPUManager()
    args.device_index = 0
    for i in flod_list:
        args_new = copy.deepcopy(args)
        args_new.flod_list = [i]
        buc.append(main(args_new))
    para_mean_result_dict = {}
    if args.subjects_type == 'inter':
        for tup in buc:
            result.acc_flod_list[tup[0]] = tup[1]
            result.subjectsScore[tup[2]] = tup[3]
    elif args.subjects_type == 'intra':
        for tup in buc:
            result.acc_flod_list[tup[0]] = tup[1]
            result.subjects_results[:, tup[2]] = tup[3]
            result.label_val[:, tup[2]] = tup[4]
    for tup in buc:
        if len(para_mean_result_dict) == 0:
            para_mean_result_dict = tup[-1]
        else:
            for k, v in tup[-1].items():
                para_mean_result_dict[k]['now_best_acc_train'] += v['now_best_acc_train']
                para_mean_result_dict[k]['now_best_acc_val'] += v['now_best_acc_val']
    for k in para_mean_result_dict.keys():
        para_mean_result_dict[k]['now_best_acc_train'] /= len(para_mean_result_dict)
        para_mean_result_dict[k]['now_best_acc_val'] /= len(para_mean_result_dict)
    json.dump({
        "para_mean_result_dict": para_mean_result_dict
    }, open(os.path.join(args.model_path, 'para_mean_result_dict.json'), 'w'))

    Utils.print_res(args, result)
    Utils.draw_res(args)

