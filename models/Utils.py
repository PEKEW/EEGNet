import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
import numpy as np
from torch.utils.data import Dataset

def getDEFreature(rawData):
    raise NotImplementedError

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

def maybeNumNodes(edgeIdx, numNodes=None):
    return edgeIdx.max().item() + 1 if numNodes is None else numNodes


def addRemainingSelfLoops(edgeIdx, edgeWeight=None, fillValue=1, numNodes=None):
    """添加自环 并返回更新的边索引和边权重

    Args:
        edgeIdx (tensor): 边索引
        edgeWeight (tensor, optional): 边权重. Defaults to None.
        fillValue (int, optional): 填充值1表示默认填充2表示增强填充. Defaults to 1.
        numNodes (int, optional): 节点数量. Defaults to None.

    Returns:
        tensor tensor: 边索引 边权重
    """
    # edgeWeight shape : (numNondes * numNodes*batchSize)
    numNodes = maybeNumNodes(edgeIdx, numNodes)
    row, col = edgeIdx
    mask = row != col
    invMask = ~mask

    loopWeight = torch.full((numNodes,), fillValue, dtype = None if edgeWeight is None else edgeWeight.dtype, device=edgeIdx.device)
    if (edgeWeight is not None):
        assert edgeWeight.numel() == edgeIdx.size(1)
        remainingEdgeWeight = edgeWeight[invMask]

        if remainingEdgeWeight.numel() > 0:
            loopWeight[row[invMask]] = remainingEdgeWeight
        
        edgeWeight = torch.cat([edgeWeight[mask], loopWeight], dim=0)
    
    loopIdx = torch.arange(0, numNodes, dtype=row.dtype, device=row.device)
    loopIdx = loopIdx.unsqueeze(0).repeat(row.size(0), 1)
    edgeIdx = torch.cat([edgeIdx[:, mask], loopIdx], dim=1)

    return edgeIdx, edgeWeight

class _SGConv(SGConv):
    def __init__(self, numFeatures, numClasses, K=1, cached=False, bias=True):
        super(_SGConv, self).__init__(numFeatures, numClasses, K, cached, bias)
        nn.init.xavier_normal_(self.lin.weight)
    
    @staticmethod
    def norm(edgeIdx, numNodes, edgeWeight, improved=False, dtype=None):
        """对图的边权重进行归一化处理

        Args:
            edgeIdx (tensor): 边索引
            numNodes (int): 节点数量
            edgeWeight (tensor): 边权重
            improved (bool, optional): 是否使用提升归一化. Defaults to False.
            dtype (edgeWeight.dtype, optional): 数据类型. Defaults to None.

        Returns:
            tensor tensor: 归一化的邻接矩阵
        """
        if edgeWeight is None:
            edgeWeight = torch.ones((edgeIdx.size(1), ), dtype=dtype, device=edgeIdx.device)
        fillValue = 1 if not improved else 2
        edgeIdx, edgeWeight = addRemainingSelfLoops(edgeIdx, edgeWeight, fillValue, numNodes)
        row, col = edgeIdx
        deg = scatter_add(torch.abs(edgeWeight), row, dim=0, dimSize=numNodes)
        degInvSqurt = deg.pow(-0.5) # 度矩阵归一化
        degInvSqurt[degInvSqurt == float('inf')] = 0 # 规范除0
        # 归一化邻接矩阵
        return edgeIdx, degInvSqurt[row] * edgeWeight * degInvSqurt[col]
    
    def forward(self, x, edgeIdx, edgeWeight=None):

        # 缓存加速
        if not self.cached or self.cachedResult is None:
            edgeIdx, norm = self.norm(edgeIdx, x.size(0), edgeWeight, dtype=x.dtype)

            for k in range(self.K):
                x =self.propagate(edgeIdx, x=x, norm=norm)
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