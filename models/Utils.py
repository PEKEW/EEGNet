import json
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from typing import List, Tuple
from itertools import product
from Datasets.Datasets import VRSicknessDataset, InterSubjectSampler, ExtraSubjectSampler, GenderSubjectSamplerMale, GenderSubjectSamplerFemale
from Datasets.DatasetsUtils import SequenceCollator
import models.trainer as Trainer

def get_data_loaders_gender(args) -> Tuple[DataLoader]:
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=['eeg'])
    def create_loader(sampler):
        return DataLoader(
            datasets,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=SequenceCollator(sequence_length=None, padding_mode='zero',include=['eeg']),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    male_loader = create_loader(GenderSubjectSamplerMale(datasets))
    female_loader = create_loader(GenderSubjectSamplerFemale(datasets))
    return male_loader, female_loader


def get_range(args):
    # todo : using json file to make grid search
    """get the range of hyperparameters
    """
    # file = open('./gridSearchConfig.json', 'r')
    # task = str(args.subjects_type) + str(args.num_classes)
    range_dict = {
        "lr": [args.lr],
        "num_hiddens": [args.num_hiddens],
        "l1_reg": [args.l1_reg],
        "l2_reg": [args.l2_reg]
    }
    return range_dict


class Results(object):
    """results of the model
    """
    def __init__(self, args):
        self.valAccfolds = np.zeros(args.n_folds)
        self.subjectsScore = np.zeros(args.n_subs)
        self.accfold_list = [0] * 10;
        if args.subjects_type == 'intra':
            self.subjectsResults = np.zeros((args.n_subs, args.sec * args.n_vids))
            self.labelVal = np.zeros((args.n_subs, args.sec * args.n_vids))


class NormalDataset(Dataset):
    def __init__(self, data, label, device):
        super(NormalDataset, self).__init__()
        self.data = data
        self.label = label
        self.device = device
    def __getitem__(self, index):
        x = np.array(self.data[index])
        y = np.array(self.label[index])
        return torch.from_numpy(x).to(self.device, dtype=torch.float32), torch.from_numpy(y).to(self.device, dtype=torch.int32)








def l1_regLoss(model, only=None, exclude=None):
    """返回sqared L1正则化损失
    """
    totalLoss = 0
    if only is None and exclude is None:
        for name, param in model.namded_parameters():
            totalLoss += torch.sum(torch.abs(param))
    elif only is not None:
        for name, param in model.namded_parameters():
            if name in only:
                totalLoss += torch.sum(torch.abs(param))
    elif exclude is not None:
        for name, param in model.namded_parameters():
            if name not in exclude:
                totalLoss += torch.sum(torch.abs(param))
    return totalLoss
def l2_regLoss(predict, label):
    if type(predict) == np.ndarry:
        numSamples = predict.shape[0]
    elif type(predict) == list:
        numSamples = len(predict)
        predict = np.array(predict)
        label = np.array(label)
    
    return np.sum(predict == label) / numSamples if numSamples > 0 else 0

class GPUManager():
    # todo check if cuda is available
    """gpu manager, list all available gpu devices, and auto choice the most free one
    """
    def __init__(self, qargs=[]):
        self.qargs = qargs
        self.gpus = self.queryGpu(qargs)
        for gpu in self.gpus:
            gpu['specified'] = False
        self.gpu_num = len(self.gpus)

    @staticmethod
    def _sortByMemory(gpus, by_size=False):
        if by_size:
            print('Sorted by free memory size')
            return sorted(gpus, key=lambda d: d['memory.free'], reverse=True)
        else:
            print('Sorted by free memory rate')
            return sorted(gpus, key=lambda d: float(d['memory.free']) / d['memory.total'], reverse=True)

    def _sortByPower(self, gpus):
        return sorted(gpus, key=self.byPower)
    

    @staticmethod
    def _sortByCustom(gpus, key, reverse=False, qargs=[]):
        if isinstance(key, str) and (key in qargs):
            return sorted(gpus, key=lambda d: d[key], reverse=reverse)
        if isinstance(key, type(lambda a: a)):
            return sorted(gpus, key=key, reverse=reverse)
        raise ValueError("The argument 'key' must be a function or a key in query args,please read the documention of nvidia-smi")
    

    def autoChoice(self, mode=0):
        for old_infos, new_infos in zip(self.gpus, self.queryGpu(self.qargs)):
            old_infos.update(new_infos)
        unspecified_gpus = [gpu for gpu in self.gpus if not gpu['specified']] or self.gpus
        if mode == 0:
            print('Choosing the GPU device has largest free memory...')
            chosen_gpu = self._sortByMemory(unspecified_gpus, True)[0]
        elif mode == 1:
            print('Choosing the GPU device has highest free memory rate...')
            chosen_gpu = self._sortByPower(unspecified_gpus)[0]
        elif mode == 2:
            print('Choosing the GPU device by custom...')
            chosen_gpu = self._sortByCustom(unspecified_gpus, key=lambda d: d['memory.free'], reverse=True)
        chosen_gpu['specified'] = True
        index = chosen_gpu['index']
        print(f'Choosing GPU device {index}...')
        return int(index)

    def queryGpu(self, qargs=[]):
        qargs =['index','gpu_name', 'memory.free', 'memory.total', 'power.draw', 'power.limit']+ qargs
        cmd = 'nvidia-smi --query-gpu={} --format=csv,noheader'.format(','.join(qargs))
        results = os.popen(cmd).readlines()
        return [self.parse(line,qargs) for line in results]

    @staticmethod
    def parse(line, qargs):
        '''
            解析一行nvidia-smi返回的csv格式文本
        '''
        numberic_args = ['memory.free', 'memory.total', 'power.draw', 'power.limit']#可计数的参数
        power_manage_enable=lambda v:(not 'Not Support' in v)#lambda表达式，显卡是否滋瓷power management（笔记本可能不滋瓷）
        to_numberic=lambda v:float(v.upper().strip().replace('MIB','').replace('W',''))#带单位字符串去掉单位
        process = lambda k,v:((int(to_numberic(v)) if power_manage_enable(v) else 1) if k in numberic_args else v.strip())
        return {k:process(k,v) for k,v in zip(qargs,line.strip().split(','))}
    
    @staticmethod
    def byPower(d):
        powerInfos = (d['power.draw'], d['power.limit'])
        if any(v==1 for v in powerInfos):
            return 1
        return float(d['power.draw']) / d['power.limit']
    

def drawRatio(model_path, csvName, figName, cls=2):
    nSub = 123
    path = os.path.join(model_path, csvName)
    data = pd.read_csv(path)
    accList = np.array(data[['0']]) * 100
    accMean = np.mean(accList)
    std = np.std(accList)
    print(figName + ' mean: %.1f' % accMean, ' std: %.1f' % std)
    plt.figure(figsize=(10, 10))
    titleName = figName + ' mean: %.1f' % accMean + ' std: %.1f' % std
    plt.title(titleName, fontsize=20, loc='center')
    xHaxis = [str(num) for num in range(1, nSub + 1 + 1)]
    y = np.vstack((accList, accMean)).flatten()
    y[:-1] = np.sort(y[:-1])
    x = np.arange(0, len(xHaxis))
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)', fontsize=30)
    plt.yticks(fontsize=25)
    plt.bar(x[:-1], y[:-1], facecolor='#D3D3D3', edgecolor='black', width=0.9, label='accuacy for each subject')
    plt.bar(x[-1] + 5, y[-1], facecolor='#696969', edgecolor='black', width=2.5, label='averaged accuracy')
    plt.errorbar(x[-1] + 5, y[-1], yerr=std, fmt='o', ecolor='black', color='#000000', elinewidth=1, capsize=2, capthick=1)

    y_ = np.ones((y.shape[0] + 0)) * 1 / int(cls) * 100
    x_ = np.arange(0, y_.shape[0])
    plt.plot(x_, y_, linestyle='dashed', color='#808080')

    plt.savefig(os.path.join(model_path, figName + '.png'))
    plt.savefig(os.path.join(model_path, figName + '.eps'), format='eps')
    plt.clf()

def get_edge_weight() -> \
    Tuple[str, List[List[int]], List[List[float]]]:
    """
    返回edge idx 和 edge weight
    edge 是二维数组，分别表示每个电极和彼此之间的连接
    edge idx 是一个二维数组 表示电极的索引
    例如 edge_idx[0] = [0,1,2,3...]
    edge_idx[1] = [0,1,2,3...]
    edge_weight 是一个二维数组 表示电极之间的连接权重
    两个电极如果是同一个 则连接权重为1
    否则为两个电极之间的距离的平方
    delta是一个常数，用于控制连接的稀疏程度
    如果两个电极之间的距离的平方小于delta，则连接权重为0
    否则为exp(-距离的平方/2) 这表示连接距离越远 权重越小 
    为什么使用指数函数可能是因为更平滑的变形
    """
    total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 Fc1 Fc2 Fc5 Fc6 Cz C3 C4 T7 T8 Cp1 Cp2 Cp5 Cp6 Pz P3 P4 P7 P8 Po3 Po4 Oz O1 O2'''.split()
    raw_pos_value = np.load('/home/pekew/code/EEGNet/models/pos.npy') * 100
    # raw_pos_value 有32个位置，最后两个是参考电极 不用考虑
    edge_pos_value = {
        name: raw_pos_value[idx]
        for idx, name in enumerate(total_part)
    }
    edge_weight = np.zeros([len(total_part), len(total_part)])
    delta = 2
    edge_index = [[], []]

    for (i, electrode1), (j, electrode2) in product(enumerate(total_part), enumerate(total_part)):
        edge_index[0].append(i)
        edge_index[1].append(j)

        if i==j:
            edge_weight[i][j] = 1
        else:
            pos1 = edge_pos_value[electrode1]
            pos2 = edge_pos_value[electrode2]
            edge_weight[i][j] = np.sum((pos1 - pos2) ** 2)

            if delta / edge_weight[i][j] > 1:
                edge_weight[i][j] = math.exp(-edge_weight[i][j] / 2)
            else:
                edge_weight[i][j] = 1e-10
    return total_part, edge_index, edge_weight

def drawRes(args):
    csvName = 'subject_%s_vids_%s_valid_%s.csv' % (args.subjects_type, str(args.n_vids), args.valid_method)
    drawRatio(args.model_path, csvName, '%s_acc_%s_%s_%s' % (args.model, args.subjects_type, str(args.n_vids), args.nowTime), cls=args.num_classes)


def printRes(args, result):
    subjectScore = result.subjectsScore
    if args.subjects_type == 'intra':
        subjectResults = result.subjectsResults
        labelVal = result.labelVal
    print('acc mean: %.3f, std: %.3f' %(np.mean(result.accfold_list), np.std(result.accfold_list)))

    if args.subjects_type == 'intra':
        subjectScore = [np.sum(subjectResults[i, :] == labelVal[i, :]) 
                    / subjectResults.shape[1] for i in range(0, args.n_subs)]
    pd.DataFrame(subjectScore).to_csv(
        os.path.join(args.model_path, 
                    'subject_%s_vids_%s_valid_%s.csv' 
                    % (args.subjects_type, str(args.n_vids), args.valid_method)
                    )
    )







def benchmark(args):
    data_root_dir = args.data_root_dir
    fold_list = args.fold_list
    n_subs = args.n_subs
    n_per = args.n_per
    band_used = args.band
    range_dict = get_range(args)
    new_args = copy.deepcopy(args)
    for fold in fold_list:
        print('fold:', fold)
        now_fold_dir = os.path.join(args.model_path, 'subject_%s_vids_%s_fold_%s_valid_%s' % 
                    (args.subjects_type, str(args.n_vids), str(fold), args.valid_method))
        os.makedirs(now_fold_dir)

        train_loader, val_loader = get_data_loaders(args, fold, mod=['eeg'])

    train_num = len(train_loader.dataset)
    val_num = len(val_loader.dataset)

    para_result_dict = {}
    best_para_dict = {}


    best_para_dict.update(
        {
            'lr': 0,
            'num_hiddens': 0,
            'l1_reg': 0,
            'l2_reg': 0,
            'num_epoch': 0
        }
    )

    bestAcc = {
        'val': 0,
        'train': 0
    }

    count = 0

    for lr, num_hiddens in zip(range_dict["lr"], range_dict["num_hiddens"]):
        for l1_reg, l2_reg in zip(range_dict["l1_reg"], range_dict["l2_reg"]):
            _statTime = time.time()
            new_args.lr = lr
            new_args.num_hiddens = num_hiddens
            new_args.l1_reg = l1_reg
            new_args.l2_reg = l2_reg

            now_para_dir = os.path.join(
                now_fold_dir, f'lr={lr}_num_hiddens={num_hiddens}_l1_reg={l1_reg}_l2_reg={l2_reg}'
            )
            os.makedirs(now_para_dir)
            mean_acc_list = {
                'val': [0 for i in range(args.num_epochs)],
                'train': [0 for i in range(args.num_epochs)]
            }

            for sub_fold in range(3):
                trainer = Trainer.get_trainer(new_args)
                startTime = time.time()
                trainer.train(train_loader, val_loader, sub_fold, now_para_dir)
                # trainer.train(dataTrain, labelTrain, dataVal, labelVal, sub_fold, now_para_dir, reload=False, ndPredict=False)
                jfile = open(now_para_dir + "/" + '_acc_and_loss.json', 'r')
                jdict = json.load(jfile)
                evalNumCorrectList = jdict['evalNumCorrectList']
                trainNumCorrectList = jdict['trainNumCorrectList']
                for i in range(args.num_epochs):
                    mean_acc_list['val'][i] += evalNumCorrectList[i]
                    mean_acc_list['train'][i] += trainNumCorrectList[i]
                endTime = time.time()

                print(f"thread id: {args.threadID}, fold: {fold}, subfold: {sub_fold},  l2_reg: {l2_reg}, bestAcc: {jdict['bestAcc']},  bestEpoch: {jdict['bestEpoch']}, time consumed: {endTime - startTime}")
            nowBestEpoch = 0
            nowBestAcc = {'val':0, 'train': 0}
            for i in range(args.num_epochs):
                mean_acc_list['val'][i] /= numTrainAndVal
                mean_acc_list['train'][i] /= 2 * numTrainAndVal
                if mean_acc_list['val'][i] > nowBestAcc['val']:
                    nowBestAcc['val'] = mean_acc_list['val'][i]
                    nowBestAcc['train'] = mean_acc_list['train'][i]
                    nowBestEpoch = i
                para_result_dict.update({
                    count: {
                        "lr": lr,
                        "num_hiddens": num_hiddens,
                        "l1_reg": l1_reg,
                        "l2_reg": l2_reg,
                        "nowBestAccTrain": nowBestAcc['train'],
                        "nowBestAccVal": nowBestAcc['val'],
                        "nowBestEpoch": nowBestEpoch
                    }
                })
                count += 1
                json.dump({
                    'fold': int(fold),
                    'nowBestAccTrain': nowBestAcc['train'],
                    'nowBestAccVal' : nowBestAcc['val'],
                    'nowBestEpoch': nowBestEpoch,
                    'lr': lr,
                    'num_hiddens': num_hiddens,
                    'l1_reg': l1_reg,
                    'l2_reg': l2_reg,
                    'timeConsumed': endTime - startTime
                }, open(now_para_dir + 
                        f'/fold_{fold}_meanAccAndLoss.json', 'w'))
            if nowBestAcc['val'] > bestAcc['val']:
                bestAcc['val'] = nowBestAcc['val']
                bestAcc['train'] = nowBestAcc['train']
                best_para_dict.update({
                    'lr': lr,
                    'num_hiddens': num_hiddens,
                    'l1_reg': l1_reg,
                    'l2_reg': l2_reg,
                    'num_epochs': nowBestEpoch
                })
        print(f'fold: {fold} choosee para: {best_para_dict}, bestAccVal: {bestAcc["val"]}, bestAccTrain: {bestAcc["train"]}')

        new_args.lr = best_para_dict['lr']
        new_args.num_hiddens = best_para_dict['num_hiddens']
        new_args.l1_reg = best_para_dict['l1_reg']
        new_args.l2_reg = best_para_dict['l2_reg']
        new_args.num_epochs = best_para_dict['num_epochs'] + 1

        trainer = trainer.getTrainer(new_args)
        startTime = time.time()

        predsTrainAndVal, predsTest = trainer.train(
            data_train_and_val, 
            label_train_and_val, 
            label_test, 
            fold,
            now_fold_dir, reload=False)
        
        endTime = time.time()
        trainAdnValAcc = np.sum(predsTrainAndVal == label_train_and_val) / len(label_train_and_val)
        testAcc = np.sum(predsTest == label_test) / len(label_test)

        print(f'--final test acc -- thread id: {args.threadID}, fold: {fold}, testAcc: {testAcc}, trainAndValAcc: {trainAdnValAcc}, time consumed: {endTime - startTime}')
        json.dump({
            'fold': int(fold),
            'trainAndValAcc': trainAdnValAcc,
            'testAcc': testAcc,
            'bestValAcc': bestAcc['val'],
            'bestParaDict': best_para_dict,
            'paraResultDict': para_result_dict,
            'timeConsumed': endTime - startTime
        }, open(now_fold_dir + '/fold_{fold}_accAndLoss.json', 'w'))

        subjectsResults = predsTest
        if args.subjects_type == 'inter':
            subjectsResults = predsTest.reshape(test_sub.shape[0], -1)
            label_test = np.array(label_test).reshape(test_sub.shape[0], -1)
            TestResult = [
                np.sum(subjectsResults[i, :] == label_test[i, :]) /
                subjectsResults.shape[1] for i in range(0, test_sub.shape[0])
            ]
            return (fold, testAcc, TestResult)
        elif args.subjects_type == 'intra':
            subjectsResults = subjectsResults.reshape(n_subs, -1)
            label_test = np.array(label_test).reshape(n_subs, -1)
            return (fold, testAcc, test_list, subjectsResults, label_test, para_result_dict)







def get_data_loaders(args, fold, mod = ['eeg']) -> Tuple[DataLoader]:
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=mod)
    all_subjects = sorted(set(sub_id for sub_id, _ in datasets.samples))
    if args.subjects_type == 'inter':
        train_sampler = InterSubjectSampler(datasets, fold, all_subjects, args.n_per, True)
        val_sampler = InterSubjectSampler(datasets, fold, all_subjects, args.n_per, False)
    else:
        train_sampler = ExtraSubjectSampler(datasets, fold, all_subjects, args.n_per, True)
        val_sampler = ExtraSubjectSampler(datasets, fold, all_subjects, args.n_per, False)
    collator = SequenceCollator(sequence_length=None, padding_mode='zero')
    train_loader = DataLoader(
        datasets,
        sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True
    )
    val_loader = DataLoader(
        datasets,
        sampler=val_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True
    )
    return train_loader, val_loader
