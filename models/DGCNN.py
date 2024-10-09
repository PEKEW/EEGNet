import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
from Utils import _SGConv


class DGCNN(nn.Module):
    def __init__(self, device, numNodes, edgeWeight, edgeIdx, numFeatures, numHiddens, numClasses, numLayers, learnEdgeWeight=True, dropout=0.5):
        """DGCNN model.

        Args:
            device (int): model device.
            numNodes (int): number of nodes in the graph.
            edgeWeight (tensor): edge matrix.
            edgeIdx (tensor): edge index.
            numFeatures (int): number of features for each node/channel.
            numHiddens (int): number of hidden dimensions.
            numClasses (int): classes number.
            numLayers (int): number of layers.
            learnEdgeWeight (bool, optional): if edge weight is learnable. Defaults to True.
            dropout (float, optional): dropout. Defaults to 0.5.
        """
        super(DGCNN, self).__init__()
        self.device = device
        self.numNodes = numNodes
        self.xs, self.ys = torch.tril_indices(self.numNodes, self.numNodes, offset=0)
        # todo 这里本身就是tensor，不用再转换，注意实现的时候
        self.edgeIdx = torch.tensor(edgeIdx)
        edgeWeight = edgeWeight.reshape(self.numNodes, self.numNodes)[self.xs, self.ys]
        self.edgeWeight = nn.Parameter(torch.Tensor(edgeWeight).float(), requires_grad=learnEdgeWeight)
        self.dropout = dropout
        self.conv1 = _SGConv(numFeatures=numFeatures, numClasses=numHiddens, K=numLayers)
        self.conv2 = nn.Conv1d(self.numNodes, 1, 1)
        self.fc = nn.Linear(numHiddens, numClasses)
    
def append(self, edgeIdx, batchSize):
    """concate a batch of graphs.

    Args:
        edgeIdx (edge tensor): edge idx of graoh.
        batchSize (int): size of one batch.

    Returns:
        edgeIdxAll: edge of concated
    """
    # 用于存储扩展后的边索引
    edgeIdxAll = torch.LongTensor(2, edgeIdx.shape[1] * batchSize)
    # 用于存储batch索引
    dataBatch = torch.LongTensor(self.numNodes * batchSize)
    for i in range(batchSize):
        edgeIdxAll[:, i*edgeIdx.shape[1]:(i+1)*edgeIdx.shape[1]] = ...
        edgeIdx + i * self.numNodes
        dataBatch[i*self.numNodes:(i+1)*self.numNodes] = i
    return edgeIdxAll.to(self.device), dataBatch.to(self.device)

def forward(self, x):
    batchSize = len(x)
    print(f"debug info: batch size {batchSize}")
    x = x.reshape(-1, x.shape[-1])
    edgeIdx, _ = self.append(self.edgeIdx, batchSize)
    edgeWeight = torch.zeros((self.numNodes, self.numNodes), device=edgeIdx.device)
    edgeWeight[self.xs.to(edgeWeight.device), self.ys.to(edgeWeight.device)] = self.edgeWeight
    # 对角化 和 reshape 重复:确保每个批次中的每个样本都使用相同的edgeweight值
    edgeWeight = edgeWeight + edgeWeight.transpose(1, 0) - torch.diag(edgeWeight.diagonal())
    edgeWeight = edgeWeight.reshape(-1).repeat(batchSize)
    # shpe: (2, self.numNodes*self.numNodes*batchSize)  edge_weight: (self.numNodes*self.numNodes*batchSize,)
    # 这样的形状包含图中所有的边 包括自环
    x = self.conv1(x, edgeIdx, edgeWeight)
    x = x.view((batchSize, self.numNodes, -1))
    x = self.conv2(x)
    x = F.relu(x.squeeze(1))
    x = self.fc(x)
    return x
