import os
import copy
import time
import random
import numpy as np
import pandas as pd
import argparse
import json
import torch
import matplotlib.pyplot as plt
import hdf5storage as hdf5
import math

# from manager_torch import GPUManager

# todo trans some func to utils.py



class Results(object):
    def __init__(self, args):
        self.valAccFlods = np.zeros(args.nFloats)
        self.subjectsScore = np.zeros(args.nSubs)
        self.accFlodList = [0] * 10;
        if args.subjectType == 'intra':
            self.subjectsResults = np.zeros((args.nSubs, args.sec * args.nVids))
            self.labelVal = np.zeros((args.nSubs, args.sec * args.nVids))

def init():
    parser = argparse.ArgumentParser(
        description = ' EEG Sickness Net'
    )
    parser.add_argument('--randSeed', default=0, type=int, help='random seed')
    parser.add_argument('--trainFlod', default='all', type=str, help='training flod, 0-9')
    parser.add_argument('--subjectsType', default='inter', type=str, help='inter or intra subject', choices=['inter', 'intra'])
    parser.add_argument('--validMehtod', default='kfold', type=str, help='kfold or leave-one-out', choices=['10-fold', 'leave-one-out'])
    parser.add_argument('--autoDeviceCount', default=5, type=int, help='num of GPU auto find and use at the same time')
    parser.add_argument('--deviceList', default='[0]', type=str, help='device list')
    parser.add_argument('--deviceIndex', default=-1, type=int, help='device index')
    parser.add_argument('--cpu', default=False, type=str, help='use cpu')
    parser.add_argument('--earlyStop', default=20, type=int, help='early stop')
    parser.add_argument('--band', default=5, type=int, help='num of bands used')
    parser.add_argument('--numNodes', default=32, type=int, help='num of nodes')
    parser.add_argument('--numEpochs', default=100, type=int, help='num of epochs')
    parser.add_argument('--l1Reg', default=0, type=float, help='l1 regularization')
    parser.add_argument('--l2Reg', default=0, type=float, help='l2 regularization')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout')
    parser.add_argument('--numHiddens', default=900, type=int, help='num of hiddens in W')
    parser.add_argument('--numLayers', default=2, type=int, help='num of layers')
    parser.add_argument('--nVids', default=24, type=int, help='num of videos')

    args = parser.parse_args()
    args.numClasses = 2
    args.cpu = True if args.cpu == 'True' else False
    args.deviceList = json.load(args.deviceLlist)
    torch.manual_seed(args.randSeed)
    torch.cuda.manual_seed(args.randSeed)
    torch.cuda.manual_seed_all(args.randSeed)
    np.random.seed(args.randSeed)
    random.seed(args.randSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.validMehtod == '10-fold':
        args.nFlods = 10
    else:
        args.nFlods = args.nSubs

    # todo
    args.nSubs = 123
    args.nPer = round(args.nSubs / args.nFlods)
    args.sec = 30

    args.l1Reg = 0.001
    args.l2Reg = 0.001

    dataRootDir = './Data/' + str(args.band) + 'bands/smooth_' + str(args.nVids)
    args.dataRootDir = dataRootDir

    nowTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    args.nowTime = nowTime

    modelPath = f'./result/_{nowTime}_{args.subjectsType}_{args.nVids}'
    if os.path.exists(modelPath) == False:
        os.makedirs(modelPath)
    args.modelPath = modelPath

    json.dump(vars(args), open(modelPath + f'/args_{nowTime}.json', 'w'))

    return args


def getRange(args):
    # file = open('./gridSearchConfig.json', 'r')
    # task = str(args.subjectsType) + str(args.numClasses)
    rangeDict = {
        "lr": [args.lr],
        "numHiddens": [args.numHiddens],
        "l1Reg": [args.l1Reg],
        "l2Reg": [args.l2Reg]
    }
    return rangeDict

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

def loadSrtDe():
    raise NotImplementedError

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
                trainer = getTrainer(newArgs)
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

        trainer = getTrainer(newArgs)
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

def getEdgeWeight(args):
    totalPart = '''Fp1 Fp2 Fz F3 F4 F7 F8 FC1 FC2 FC5 FC6 Cz C3 C4 T7 T8 CP1 CP2 CP5 CP6 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2'''.split()
    edgePosValue = np.load('./Models/pos.npy') * 100
    edegeWeight = np.zeros([len(totalPart), len(totalPart)])
    delta = 2 # make the proportion fo non negligible connection exactly 20%
    edgeIndex = [[], []]
    for i in range(len((totalPart))):
        for j in range(len(totalPart)):
            edgeIndex[0].append(i)
            edgeIndex[1].append(j)
            if i == j:
                edegeWeight[i][j] = 1
            else:
                edegeWeight[i][j] = np.sum(
                    [(edgePosValue[i][k] - edgePosValue[j][k]) ** 2 for k in range(2)]
                )
                if delta / edegeWeight[i][j] > 1:
                    edegeWeight[i][j] = math.exp(-edegeWeight[i][j]/2)
                else:
                    edegeWeight[i][j] = 0
    return edgeIndex, edegeWeight


def getTrainer(args):
    if args.cpu == True:
        deviceUsing = torch.device('cpu')
    else:
        deviceUsing = torch.device('cuda:%d' % args.deviceIndex)
    edgeIndex, edgeWeight = getEdgeWeight(args)
    return DGCNNTrainer(
            edgeIndex = edgeIndex,
            edgeWeight = edgeWeight,
            numClasses=args.numClasses,
            device=deviceUsing,
            numHiddens = args.numHiddens,
            numLayers = args.numLayers,
            dropout = args.dropout,
            barchSize = args.batchSize,
            lr = args.lr,
            l2Reg = args.l2Reg,
            numEpochs = args.numEpochs
        )


def trainValSplit(args, flod, subFlod, dataTrainAndVal, labTrainAndVal, testList):
    raise NotImplementedError

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


def drawRes(args):
    csvName = 'subject_%s_vids_%s_valid_%s.csv' % (args.subjectsType, str(args.nVids), args.validMethod)
    drawRatio(args.modelPath, csvName, '%s_acc_%s_%s_%s' % (args.model, args.subjectsType, str(args.nVids), args.nowTime), cls=args.numClasses)
def main(args):
    return benchmark(args)

if __name__ == '__main__':
    args = init()
    if args.trainFlod == 'all':
        flodList = np.arange(0, args.nFlods)
    else:
        flodList = [int(args.trainFlod)]
    result = Results(args)
    buc = []
    autoChoiceMod = 1
    gm = GPUManager()
    args.threadID = 0
    args.deviceIndex = gm.auto_choice(autoChoiceMod)
    for i in flodList:
        argsNew = copy.deepcopy(args)
        argsNew.flodList = [i]
        buc.append(main(argsNew))
    paraMeanResultDict = {}
    if args.subjectsType == 'inter':
        for tup in buc:
            result.accFlodList[tup[0]] = tup[1]
            result.subjectsScore[tup[2]] = tup[3]
    elif args.subjectsType == 'intra':
        for tup in buc:
            result.accFlodList[tup[0]] = tup[1]
            result.subjectsResults[:, tup[2]] = tup[3]
            result.labelVal[:, tup[2]] = tup[4]
    for tup in buc:
        if len(paraMeanResultDict) == 0:
            paraMeanResultDict = tup[-1]
        else:
            for k, v in tup[-1].items():
                paraMeanResultDict[k]['nowBestAccTrain'] += v['nowBestAccTrain']
                paraMeanResultDict[k]['nowBestAccVal'] += v['nowBestAccVal']
    for k in paraMeanResultDict.keys():
        paraMeanResultDict[k]['nowBestAccTrain'] /= len(paraMeanResultDict)
        paraMeanResultDict[k]['nowBestAccVal'] /= len(paraMeanResultDict)
    json.dump({
        "paraMeanResultDict": paraMeanResultDict
    }, open(os.path.join(args.modelPath, 'paraMeanResultDict.json'), 'w'))

    printRes(args, result)
    drawRes(args)


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

class DGCNNTrainer(object):
    raise NotImplementedError