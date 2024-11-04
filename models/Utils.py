import json
import hdf5storage as hdf5
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from typing import List

def getDEFreature(rawData):
    raise NotImplementedError


def getRange(args):
    """get the range of hyperparameters / todo : using json file to make grid search
    """
    # file = open('./gridSearchConfig.json', 'r')
    # task = str(args.subjectsType) + str(args.numClasses)
    rangeDict = {
        "lr": [args.lr],
        "numHiddens": [args.numHiddens],
        "l1Reg": [args.l1Reg],
        "l2Reg": [args.l2Reg]
    }
    return rangeDict


class Results(object):
    """results of the model
    """
    def __init__(self, args):
        self.valAccFlods = np.zeros(args.nFloats)
        self.subjectsScore = np.zeros(args.nSubs)
        self.accFlodList = [0] * 10;
        if args.subjectType == 'intra':
            self.subjectsResults = np.zeros((args.nSubs, args.sec * args.nVids))
            self.labelVal = np.zeros((args.nSubs, args.sec * args.nVids))


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

def maybenum_nodes(edge_idx, num_nodes=None):
    return edge_idx.max().item() + 1 if num_nodes is None else num_nodes


def addRemainingSelfLoops(edge_idx, edge_weight=None, fillValue=1, num_nodes=None):
    """添加自环 并返回更新的边索引和边权重

    Args:
        edge_idx (tensor): 边索引
        edge_weight (tensor, optional): 边权重. Defaults to None.
        fillValue (int, optional): 填充值1表示默认填充2表示增强填充. Defaults to 1.
        num_nodes (int, optional): 节点数量. Defaults to None.

    Returns:
        tensor tensor: 边索引 边权重
    """
    # edge_weight shape : (numNondes * num_nodes*batchSize)
    num_nodes = maybenum_nodes(edge_idx, num_nodes)
    row, col = edge_idx
    mask = row != col
    invMask = ~mask

    loopWeight = torch.full((num_nodes,), fillValue, dtype = None if edge_weight is None else edge_weight.dtype, device=edge_idx.device)
    if (edge_weight is not None):
        assert edge_weight.numel() == edge_idx.size(1)
        remainingedge_weight = edge_weight[invMask]

        if remainingedge_weight.numel() > 0:
            loopWeight[row[invMask]] = remainingedge_weight
        
        edge_weight = torch.cat([edge_weight[mask], loopWeight], dim=0)
    
    loopIdx = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loopIdx = loopIdx.unsqueeze(0).repeat(row.size(0), 1)
    edge_idx = torch.cat([edge_idx[:, mask], loopIdx], dim=1)

    return edge_idx, edge_weight

class _SGConv(SGConv):
    def __init__(self, numFeatures, numClasses, K=1, cached=False, bias=True):
        super(_SGConv, self).__init__(numFeatures, numClasses, K, cached, bias)
        nn.init.xavier_normal_(self.lin.weight)
    
    @staticmethod
    def norm(edge_idx, num_nodes, edge_weight, improved=False, dtype=None):
        """对图的边权重进行归一化处理

        Args:
            edge_idx (tensor): 边索引
            num_nodes (int): 节点数量
            edge_weight (tensor): 边权重
            improved (bool, optional): 是否使用提升归一化. Defaults to False.
            dtype (edge_weight.dtype, optional): 数据类型. Defaults to None.

        Returns:
            tensor tensor: 归一化的邻接矩阵
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_idx.size(1), ), dtype=dtype, device=edge_idx.device)
        fillValue = 1 if not improved else 2
        edge_idx, edge_weight = addRemainingSelfLoops(edge_idx, edge_weight, fillValue, num_nodes)
        row, col = edge_idx
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dimSize=num_nodes)
        degInvSqurt = deg.pow(-0.5) # 度矩阵归一化
        degInvSqurt[degInvSqurt == float('inf')] = 0 # 规范除0
        # 归一化邻接矩阵
        return edge_idx, degInvSqurt[row] * edge_weight * degInvSqurt[col]
    
    def forward(self, x, edge_idx, edge_weight=None):

        # 缓存加速
        if not self.cached or self.cachedResult is None:
            edge_idx, norm = self.norm(edge_idx, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x =self.propagate(edge_idx, x=x, norm=norm)
            self.cachedResult = x
        
        return self.lin(self.cachedResult)
    
    def message(self, xJ, norm):
        return norm.view(-1, 1) * xJ

def l1RegLoss(model, only=None, exclude=None):
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
def l2RegLoss(predict, label):
    if type(predict) == np.ndarry:
        numSamples = predict.shape[0]
    elif type(predict) == list:
        numSamples = len(predict)
        predict = np.array(predict)
        label = np.array(label)
    
    return np.sum(predict == label) / numSamples if numSamples > 0 else 0

def loadData(data):
    raise NotImplementedError



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
    

def drawRatio(modelPath, csvName, figName, cls=2):
    nSub = 123
    path = os.path.join(modelPath, csvName)
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

    plt.savefig(os.path.join(modelPath, figName + '.png'))
    plt.savefig(os.path.join(modelPath, figName + '.eps'), format='eps')
    plt.clf()

def loadSrtDe():
    raise NotImplementedError


def get_edge_weight(args) -> List[List[int], List[int]], List[List[float], List[float]]:
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
    # todo total part 是30个电极的名字
    # todo 这个pos value 是用索引直接表示的 应该改成字典格式 or something else带语意的 更准确
    total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 FC1 FC2 FC5 FC6 Cz C3 C4 T7 T8 CP1 CP2 CP5 CP6 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2'''.split()
    edge_pos_value = np.load('./Models/pos.npy') * 100
    edge_weight = np.zeros([len(total_part), len(total_part)])
    delta = 2 # make the proportion fo non negligible connection exactly 20%
    edge_index = [[], []]
    for i in range(len((total_part))):
        for j in range(len(total_part)):
            edge_index[0].append(i)
            edge_index[1].append(j)
            if i == j:
                edge_weight[i][j] = 1
            else:
                edge_weight[i][j] = np.sum(
                    [(edge_pos_value[i][k] - edge_pos_value[j][k]) ** 2 for k in range(2)]
                )
                if delta / edge_weight[i][j] > 1:
                    edge_weight[i][j] = math.exp(-edge_weight[i][j]/2)
                else:
                    # todo 很小的但不算1
                    edge_weight[i][j] = 0
    return edge_index, edge_weight

def drawRes(args):
    csvName = 'subject_%s_vids_%s_valid_%s.csv' % (args.subjectsType, str(args.nVids), args.validMethod)
    drawRatio(args.modelPath, csvName, '%s_acc_%s_%s_%s' % (args.model, args.subjectsType, str(args.nVids), args.nowTime), cls=args.numClasses)


def printRes(args, result):
    subjectScore = result.subjectsScore
    if args.subjectsType == 'intra':
        subjectResults = result.subjectsResults
        labelVal = result.labelVal
    print('acc mean: %.3f, std: %.3f' %(np.mean(result.accFlodList), np.std(result.accFlodList)))

    if args.subjectsType == 'intra':
        subjectScore = [np.sum(subjectResults[i, :] == labelVal[i, :]) 
                    / subjectResults.shape[1] for i in range(0, args.nSubs)]
    pd.DataFrame(subjectScore).to_csv(
        os.path.join(args.modelPath, 
                    'subject_%s_vids_%s_valid_%s.csv' 
                    % (args.subjectsType, str(args.nVids), args.validMethod)
                    )
    )


def benchmark(args):
    dataRootDir = args.dataRootDir
    flodList = args.flodList
    nSubs = args.nSubs
    nPer = args.nPer
    bandUsed = args.band
    rangeDict = getRange(args)
    newArgs = copy.deepcopy(args)
    for flod in flodList:
        print('flod:', flod)
        nowFlodDir = os.path.join(args.modelPath, 'subject_%s_vids_%s_flod_%s_valid_%s' % 
                    (args.subjectsType, str(args.nVids), str(flod), args.validMethod))
        os.makedirs(nowFlodDir)

        dataTrainAndVal, labelTrainAndVal, dataTest, labelTest, testSub, testList = dataPrepare(args, flod)

    numTrainAndVal = dataTrainAndVal.shape[0]
    numTest = dataTest.shape[0]

    paraResultDict = {}
    bestParaDict = {}


    bestParaDict.update(
        {
            'lr': 0,
            'numHiddens': 0,
            'l1Reg': 0,
            'l2Reg': 0,
            'numEpoch': 0
        }
    )

    bestAcc = {
        'val': 0,
        'train': 0
    }

    count = 0

    for lr, numHiddens in zip(rangeDict["lr"], rangeDict["numHiddens"]):
        for l1Reg, l2Reg in zip(rangeDict["l1Reg"], rangeDict["l2Reg"]):
            _statTime = time.time()
            newArgs.lr = lr
            newArgs.numHiddens = numHiddens
            newArgs.l1Reg = l1Reg
            newArgs.l2Reg = l2Reg

            nowParaDir = os.path.join(
                nowFlodDir, f'lr={lr}_numHiddens={numHiddens}_l1Reg={l1Reg}_l2Reg={l2Reg}'
            )
            os.makedirs(nowParaDir)
            meanAccList = {
                'val': [0 for i in range(args.numEpoch)],
                'train': [0 for i in range(args.numEpoch)]
            }

            for subFlod in range(3):
                dataTrain, labelTrain, dataVal, labelVal = trainValSplit(
                    args, flod, subFlod, dataTrainAndVal, labelTrainAndVal, testList
                )
                trainer = trainer.getTrainer(newArgs)
                startTime = time.time()
                trainer.train(dataTrain, labelTrain, dataVal, labelVal, subFlod, nowParaDir, reload=False, ndPredict=False)
                jfile = open(nowParaDir + "/" + '_acc_and_loss.json', 'r')
                jdict = json.load(jfile)
                evalNumCorrectList = jdict['evalNumCorrectList']
                trainNumCorrectList = jdict['trainNumCorrectList']
                for i in range(args.numEpoch):
                    meanAccList['val'][i] += evalNumCorrectList[i]
                    meanAccList['train'][i] += trainNumCorrectList[i]
                endTime = time.time()

                print(
                    f'thread id: {args.threadID}, 
                    flod: {flod}, subFlod: {subFlod}, 
                    l2Reg: {l2Reg}, bestAcc: {jdict['bestAcc']}, 
                    bestEpoch: {jdict['bestEpoch']}, 
                    time consumed: {endTime - startTime}'
                    )
            nowBestEpoch = 0
            nowBestAcc = {'val':0, 'train': 0}
            for i in range(args.numEpoch):
                meanAccList['val'][i] /= numTrainAndVal
                meanAccList['train'][i] /= 2 * numTrainAndVal
                if meanAccList['val'][i] > nowBestAcc['val']:
                    nowBestAcc['val'] = meanAccList['val'][i]
                    nowBestAcc['train'] = meanAccList['train'][i]
                    nowBestEpoch = i
                paraResultDict.update({
                    count: {
                        "lr": lr,
                        "numHiddens": numHiddens,
                        "l1Reg": l1Reg,
                        "l2Reg": l2Reg,
                        "nowBestAccTrain": nowBestAcc['train'],
                        "nowBestAccVal": nowBestAcc['val'],
                        "nowBestEpoch": nowBestEpoch
                    }
                })
                count += 1
                json.dump({
                    'flod': int(flod),
                    'nowBestAccTrain': nowBestAcc['train'],
                    'nowBestAccVal' : nowBestAcc['val'],
                    'nowBestEpoch': nowBestEpoch,
                    'lr': lr,
                    'numHiddens': numHiddens,
                    'l1Reg': l1Reg,
                    'l2Reg': l2Reg,
                    'timeConsumed': endTime - startTime
                }, open(nowParaDir + 
                        f'/flod_{flod}_meanAccAndLoss.json', 'w'))
            if nowBestAcc['val'] > bestAcc['val']:
                bestAcc['val'] = nowBestAcc['val']
                bestAcc['train'] = nowBestAcc['train']
                bestParaDict.update({
                    'lr': lr,
                    'numHiddens': numHiddens,
                    'l1Reg': l1Reg,
                    'l2Reg': l2Reg,
                    'numEpoch': nowBestEpoch
                })
        print(f'flod: {flod} choosee para: {bestParaDict}, bestAccVal: {bestAcc["val"]}, bestAccTrain: {bestAcc["train"]}')

        newArgs.lr = bestParaDict['lr']
        newArgs.numHiddens = bestParaDict['numHiddens']
        newArgs.l1Reg = bestParaDict['l1Reg']
        newArgs.l2Reg = bestParaDict['l2Reg']
        newArgs.numEpoch = bestParaDict['numEpoch'] + 1

        trainer = trainer.getTrainer(newArgs)
        startTime = time.time()

        predsTrainAndVal, predsTest = trainer.train(
            dataTrainAndVal, 
            labelTrainAndVal, 
            labelTest, 
            flod,
            nowFlodDir, reload=False)
        
        endTime = time.time()
        trainAdnValAcc = np.sum(predsTrainAndVal == labelTrainAndVal) / len(labelTrainAndVal)
        testAcc = np.sum(predsTest == labelTest) / len(labelTest)

        print(f'--final test acc -- thread id: {args.threadID}, flod: {flod}, testAcc: {testAcc}, trainAndValAcc: {trainAdnValAcc}, time consumed: {endTime - startTime}')
        json.dump({
            'flod': int(flod),
            'trainAndValAcc': trainAdnValAcc,
            'testAcc': testAcc,
            'bestValAcc': bestAcc['val'],
            'bestParaDict': bestParaDict,
            'paraResultDict': paraResultDict,
            'timeConsumed': endTime - startTime
        }, open(nowFlodDir + '/flod_{flod}_accAndLoss.json', 'w'))

        subjectsResults = predsTest
        if args.subjectsType == 'inter':
            subjectsResults = predsTest.reshape(testSub.shape[0], -1)
            labelTest = np.array(labelTest).reshape(testSub.shape[0], -1)
            TestResult = [
                np.sum(subjectsResults[i, :] == labelTest[i, :]) /
                subjectsResults.shape[1] for i in range(0, testSub.shape[0])
            ]
            return (flod, testAcc, TestResult)
        elif args.subjectsType == 'intra':
            subjectsResults = subjectsResults.reshape(nSubs, -1)
            labelTest = np.array(labelTest).reshape(nSubs, -1)
            return (flod, testAcc, testList, subjectsResults, labelTest, paraResultDict)



def dataPrepare(args, fold):
    dataRootDir = args.dataRootDir
    flodList = args.flodList
    nSubs = args.nSubs
    nPer = args.nPer
    bandUsed = args.band

    dataDir = os.path.join(dataRootDir, 'test_flod%d.mat' % fold)
    # todo what is de_lds
    data = hdf5.loadmat(dataDir)['de_lds']
    data, labelRepeat, nSamples = loadSrtDe(
        data, True, False, 1, args.labelType
    )
    featureShape = int(data.shape[-1] / 30)
    # label shape: 720 or 840
    # 720 = 24 * 30
    # 840 = 28 * 30
    # 24\28 is the number of videos and 30 is sample rate
    # data shape : (123, 720, 120 or 150 or 255 * 30)
    # 720 = 24*30
    # 120 = 4*30
    # 30 is channel num
    # 4 is band num

    valSub = None
    valList = None
    if args.subjectType == 'inter':
        if fold < args.nFlods - 1:
            valSub = np.arange(nPer * fold, nPer * (fold + 1))
        else:
            valSub = np.arange(nPer * fold, nSubs)
        trainSub = np.array(list(set(np.arange(nSubs)) - set(valSub)))
        dataTrain = data[list(trainSub), :, :].reshape(-1, featureShape, 30).transpose([0, 2, 1])
        dataVal = data[list(valSub), :, :].reshape(-1, featureShape, 30).transpose([0, 2, 1])
        labelTrain = np.tile(labelRepeat, len(trainSub))
        labelVal = np.tile(labelRepeat, len(valSub))
    else:
        valSeconds = 30 / args.nFolds
        trainSeconds = 30 - valSeconds
        dataList = np.arange(0, len(labelRepeat))
        valListStart = np.arange(0, len(labelRepeat), 30) + int(valSeconds * fold)
        valList = valListStart.copy()
        for sec in range(1, int(valSeconds)):
            valList = np.concatenate((valList, valListStart + sec)).astype(int)
        trainList = np.array(list(set(dataList) - set(valList))).astype(int)
        dataTrain = data[:, list(trainList), :].reshape(-1, featureShape, 30).transpose([0, 2, 1])
        labelTrain = np.tile(labelRepeat[trainList], nSubs)
        labelVal = np.tile(np.array(labelRepeat)[valList], nSubs)
    return dataTrain, labelTrain, dataVal, labelVal,  valSub, valList


def trainValSplit(args, flod, subFlod, dataTrainAndVal, labTrainAndVal, testList):
    raise NotImplementedError