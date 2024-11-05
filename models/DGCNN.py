import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
from Utils import _SGConv

# 用于eegdata的gnn
# eegdata数据维度不算batch：(30， 250)

class DGCNN(nn.Module):
    def __init__(self, device, num_nodes, edge_weight, edge_idx, num_features, num_hiddens, num_classes, num_layers, learnable_edge_weight=True, dropout=0.5):
        """DGCNN model
        Args:
            device (int): model device.
            num_nodes (int): number of nodes in the graph.
            edge_weight (tensor): edge matrix.
            edge_idx (tensor): edge index.
            numFeatures (int): number of features for each node/channel.
            num_hiddens (int): number of hidden dimensions.
            num_classes (int): classes number.
            num_layers (int): number of layers.
            learnedge_weight (bool, optional): if edge weight is learnable. Defaults to True.
            dropout (float, optional): dropout. Defaults to 0.5.
        """
        super(DGCNN, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        self.edge_idx = edge_idx
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys]
        self.edge_weight = nn.Parameter(torch.Tensor(edge_weight).float(), requires_grad=learnable_edge_weight)
        self.dropout = dropout
        self.conv1 = _SGConv(numFeatures=num_features, num_classes=num_hiddens, K=num_layers)
        self.conv2 = nn.Conv1d(self.num_nodes, 1, 1)
        self.fc = nn.Linear(num_hiddens, num_classes)
    
def append(self, edge_idx, batch_size):
    """concate a batch of graphs.

    Args:
        edge_idx (edge tensor): edge idx of graoh.
        batch_size (int): size of one batch.

    Returns:
        edge_idxAll: edge of concated
    """
    # edge_idx 的的形状： [[0,1,2,...], [0,1,2,...]]
    # 扩充后的形状： [[0,1,2,...,0,1,2,...], [0,1,2,...,0,1,2,...]]
    # 用于存储扩展后的边索引
    edge_idx_all = torch.LongTensor(2, edge_idx.shape[1] * batch_size)
    # 用于存储batch索引
    data_batch = torch.LongTensor(self.num_nodes * batch_size)
    for i in range(batch_size):
        edge_idx_all[:, i*edge_idx.shape[1]:(i+1)*edge_idx.shape[1]] = ...
        edge_idx + i * self.num_nodes
        data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
    return edge_idx_all.to(self.device), data_batch.to(self.device)

def forward(self, x):
    batch_size = x.size(0)
    edge_idx_batch, _ = self.append(self.edge_idx, batch_size)
    edge_weight_batch = self.edge_weight.repeat(batch_size)
    x = x.reshape(-1, x.size(-1))
    x = self.conv1(x, edge_idx_batch, edge_weight_batch)
    x = F.relu(x)
    x = F.dropout(x, p=self.dropout, training=self.training)
    x = x.reshape(batch_size, self.num_nodes, -1)
    x = self.conv2(x)
    x = x.squeeze(1)
    x = self.fc(x)
    return x
