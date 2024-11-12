import torch
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from typing import List, Tuple
from itertools import product
from Datasets.Datasets import VRSicknessDataset, GenderSubjectSamplerMale, GenderSubjectSamplerFemale
from Datasets.DatasetsUtils import SequenceCollator
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine
from torch.nn import functional as F

def normalize_matrix(m: torch.Tensor, symmetry: bool=True) -> torch.Tensor:
    m = F.relu(m) # 通过relu确保所有值都是正数
    if symmetry:
        m = m + torch.transpose(m, 0, 1)
    d = torch.sum(m, dim=1)
    d = 1 / torch.sqrt(d + 1e-10)
    d = torch.diag_embed(d)
    # 计算标准化后的矩阵：D^(-1/2) A D^(-1/2)
    l = torch.matmul(torch.matmul(d, m), m)
    return l


# todo 移动到 datasetsutils
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


def draw_ratio(model_path, csvName, figName, cls=2):
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
def get_edge_weight() -> \
    Tuple[str, List[List[int]], List[List[float]]]:
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
                pos1 = edge_pos_value[electrode1]
                pos2 = edge_pos_value[electrode2]
                dist = np.sum((pos1 - pos2) ** 2)
                edge_weight[i][j] = math.exp(-dist / (2 * delta))  # 使用高斯核
    return total_part, edge_index, edge_weight

def draw_res(args):
    csvName = 'subject_%s_vids_%s_valid_%s.csv' % (args.subjects_type, str(args.n_vids), args.valid_method)
    draw_ratio(args.model_path, csvName, '%s_acc_%s_%s_%s' % (args.model, args.subjects_type, str(args.n_vids), args.nowTime), cls=args.num_classes)


def print_res(args, result):
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