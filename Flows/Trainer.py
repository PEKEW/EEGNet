import time
import os
import math
import torch
import torch.nn as nn
import inspect
import numpy as np
from models.Utils import l1RegLoss, l2RegLoss, NormalDataset
from torch.utils.data import DataLoader
from torch.utils.data import DataSet

class Trainer(object):
    def __init__(self, modelF, num_nodes, numHiddens=400, numClasses=2,
                batchSize=256, numEpoch=50, lr=0.005, l1Reg=0, l2Reg=0, dp=0.5,
                earlyStop=20, optimizer='Adam', device=torch.device('cpu'),
                extention: dict=None):
        self.nunmNodes = num_nodes
        self.numHiddens = numHiddens
        self.numEpoch = numEpoch
        self.numClasses = numClasses
        self.batchSize = batchSize
        self.lr = lr
        self.l1Reg = l1Reg
        self.l2Reg = l2Reg
        self.dp = dp
        self.earlyStop = earlyStop
        self.optimizer = optimizer
        self.device = device
        self.extention = extention
        self.batchSize = batchSize
        self.modelConfigParamName = inspect.getfullargspec(modelF.__int__F).args
        self.modelF = modelF
        self.modelName = modelF.__name__.split('_')[0]
        self.earlyStop = earlyStop
        if extention is not None:
            self.__dict__.update(extention)
        self.trainerConfigParam = self.__dict__.copy()
        self.model = None
        self.numFeatures = None
        self.modelConfigParam = dict()
    def doEpoch(self, dataLoader, mode='eval', epochNum=None, retPredict=False):
        self.model = self.model.train() if mode == 'train' else self.model.eval()
        epochLoss = {
            'EntropyLoss': 0,
            'L1Loss': 0,
            'L2Loss': 0,
            'TotalLoss': 0,
        }
        totalSamples = 0
        epochMetrics = {}

        numCorrectPredict = 0
        numCorrectDomainPredict = 0

        p = epochNum/(self.numEpoch - 1) if self.numEpoch != 1 else 1

        beta = 2 / (1+math.exp(-10*p)) - 1
        lossModel2 = nn.CrossEntropyLoss()
        totalClassPredictions = []
        for i, batch in enumerate(dataLoader):
            print(f"debug info: batch {i}")
            X, Y = batch
            numSamples = X.shape[0]
            predictions = self.model(X)
            EntropyLoss = self.lossModule(predictions.float(), 
                                torch.Tensor(Y.float().long()).to(self.device))
            classPredict = predictions.argmax(axis=-1).cpu().detach().numpy()
            if retPredict:
                totalClassPredictions += [item for item in classPredict]
            
            Y = Y.cpu().detach().numpy()
            numCorrectPredict += np.sum(classPredict == Y)
            L1Loss = self.l1Reg * l1RegLoss(self.model, only=['edge_weight'])
            L2Loss = self.l2Reg * l2RegLoss(self.model, exclude=['edge_weight'])
            loss = EntropyLoss + L1Loss + L2Loss

            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 20)
                self.optimizer.step()
            with torch.no_grad():
                totalSamples += numSamples
                epochLoss['EntropyLoss'] += EntropyLoss.item()
                epochLoss['L1Loss'] += L1Loss.item()
                epochLoss['L2Loss'] += L2Loss.item()
        epochMetrics['epooch'] = epochNum
        epochMetrics['loss'] = ...
        (epochLoss['L2Loss'] + epochLoss['L1Loss'] + epochLoss['EntropyLoss']).item() 
        ... / (totalSamples / self.batchSize)
        epochMetrics['numCurrect'] = numCorrectDomainPredict
        epochMetrics['acc'] = numCorrectPredict / totalSamples

        return totalClassPredictions if retPredict else epochMetrics

    def dataPrepareTrainOnly(self, trainData, trainLabel, validData=None, matTrain=None, nunFreq=None):
        labelClass = set(trainLabel)
        assert (len(labelClass) == self.numClasses)
        trainData = NormalDataset(trainData, trainLabel, self.device)
        trainLoader = DataLoader(dataset = trainData, batchSize = self.batchSize, shuffle=True)
        return trainLoader

    def dataPrepare(self, trainData, trainLabel, validData=None, validLabel=None, matTrain=None, numFreq=None):
        labelClass = set(trainLabel)
        assert (len(labelClass) == self.numClasses)
        trainDataset = NormalDataset(trainData, trainLabel, self.device)
        validDataset = NormalDataset(validData, validLabel, self.device)
        trainLoader = DataLoader(dataset=trainDataset, batchSize=self.batchSize, shuffle=True)
        validLoader = DataLoader(dataset=validDataset, batchSize=self.batchSize, shuffle=False)
        return trainLoader, validLoader
    
    def trainAndEval(self, trainData, trainLabel, validData, ValidLabel, numFreq=None, reload=True, ndPredict=True, trainLog=False):
        self.numFeatures = trainData.shape[-1]
        self.trainerConfigParam['numFeatures'] = self.numFeatures
        for k, v in self.__dict__.items():
            if k in self.modelConfigParamName:
                self.modelConfigParamx.update({k: v})
        self.trainerConfigParam['modelConfigParam'] = self.modelConfigParam
        self.model = self.modelF(**self.modelConfigParam).to(self.device)
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        bsetAcc = 0
        earlyStop = self.earlyStop
        if earlyStop is None:
            earlyStop = self.numEpoch
        earlyStopNum = 0
        bestEpoch = -1
        evalAccList = []
        trainAccList = []
        evalLossList = []
        trainNumCorrectList = []
        evalNumCorrectList = []
        trainLossList = []
        for i in range(self.numEpoch):
            timeStart = time.time()
            trainMetric = self.doEpoch(trainData, 'train', i)
            evalMetric = self.doEpoch(validData, 'eval', i)
            timeEnd = time.time()
            timeCost = timeEnd - timeStart

            if trainLog:
                print(f"Epoch {i} train loss: {trainMetric['loss']:.4f} acc: {trainMetric['acc']:.4f} eval loss: {evalMetric['loss']:.4f} acc: {evalMetric['acc']:.4f} time: {timeCost:.2f}")
            evalAccList.append(evalMetric['acc'])
            trainAccList.append(trainMetric['acc'])
            evalLossList.append(evalMetric['loss'])
            trainLossList.append(trainMetric['loss'])
            trainNumCorrectList.append(trainMetric['numCorrect'])
            evalNumCorrectList.append(evalMetric['numCorrect'])
            if evalMetric['acc'] > bestAcc:
                bestAcc = evalMetric['acc']
                bestEpoch = i
                earlyStopNum = 0
            else:
                earlyStopNum += 1
        
        self.evalAccList = evalAccList
        self.trainAccList = trainAccList

    def trainOnly(self, trainData, trainLabel, validData=None, matTrain = None, numFreq=None, trainLog=False):
        self.numFeatures = trainData.shape[-1]
        self.trainerConfigParam['numFeatures'] = self.numFeatures
        for k, v in self.__dict__.items():
            if k in self.modelConfigParamName:
                self.modelConfigParam.update({k: v})
        self.trainerConfigParam['modelConfigParam'] = self.modelConfigParam
        self.model = self.modelF(**self.modelConfigParam).to(self.device)
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        
        trainLoader = self.dataPrepareTrainOnly(trainData, trainLabel)

        trainAccList = []
        trainNumCorrectList = []
        trainLossList = []

        for i in range(self.numEpoch):
            print(f"Epoch {i}")
            trainMetric = self.doEpoch(trainLoader, 'train', i)
            trainAccList.append(trainMetric['acc'])
            trainLossList.append(trainMetric['loss'])
            trainNumCorrectList.append(trainMetric['numCorrect'])
        self.trainAccList = trainAccList
        return trainAccList
    
    def predict(self, data, adjMat = None):
        if self.model is None:
            raise ValueError("Model not trained yet")
        data = torch.from_numpy(data).to(self.device, dtype=torch.float32)
        self.model = self.model.eval()
        totalClassPredictions = []
        with torch.no_grad():
            if data.shape[0] < 128:
                predictions = self.model(data)
                classPredict = predictions.argmax(axis=-1)
                classPredict = classPredict.cpu().detach().numpy()
                totalClassPredictions += [item for item in classPredict]
            else:
                for i in range(0, data.shape[0], 128):
                    if i + 128 < data.shape[0]:
                        predictions = self.model(data[i:i+128, :, :])
                    else:
                        predictions = self.model(data[i:, :, :])
        return np.array(totalClassPredictions)
    
    def save(self, path, name = 'bsetModelDic.pkl'):
        if self.model is None:
            raise ValueError("Model not trained yet")
        if not os.path.exist(path):
            os.makedirs(path)
        modelDict = {
            'state_dict': self.model.state_dict(),
            'configs': self.trainerConfigParam,
        }
        torch.save(modelDict, os.path.join(path, name))
    
    def load(self, path, name = 'bsetModelDic.pkl'):
        self.model = None
        modelDic = torch.load(os.path.join(path, name), map_location='cpu')
        self.__dict__.update(modelDic['configs'])
        self.trainerConfigParam = self.__dict__.copy()
        self.model = self.modelF(**self.modelConfigParam)
        self.model.load_state_dict(modelDic['state_dict'])
        self.model = self.model.to(self.device)