import os
import copy
import time
import random
import numpy as np
import argparse
import json
import torch
import Utils
from Utils import Results, GPUManager

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
    parser.add_argument('--num_nodes', default=32, type=int, help='num of nodes')
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


def main(args):
    return Utils.benchmark(args)

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

    Utils.printRes(args, result)
    Utils.drawRes(args)

