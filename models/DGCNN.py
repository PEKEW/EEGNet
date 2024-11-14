import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
import warnings
from models.Utils import normalize_matrix


class DeprecatedConvDescriptor:
    def __init__(self, _layer):
        self.conv_layer = _layer

    def __get__(self, instance, owner):
        warnings.warn(
            "The member is deprecated and will be removed in future versions. ",
            DeprecationWarning,
            stacklevel=2
        )
        return self.conv_layer



def maybe_num_nodes(edge_idx, num_nodes=None):
    return edge_idx.max().item() + 1 if num_nodes is None else num_nodes

def add_remaining_self_loops(edge_idx, edge_weight=None, fillValue=1, num_nodes=None):
    """添加自环 并返回更新的边索引和边权重

    Args:
        edge_idx (tensor): 边索引
        edge_weight (tensor, optional): 边权重. Defaults to None.
        fillValue (int, optional): 填充值1表示默认填充2表示增强填充. Defaults to 1.
        num_nodes (int, optional): 节点数量. Defaults to None.

    Returns:
        tensor tensor: 边索引 边权重
    """
    # edge_weight shape : (numNondes * num_nodes*batch_size)
    num_nodes = maybe_num_nodes(edge_idx, num_nodes)
    row, col = edge_idx
    mask = row != col
    inv_mask = ~mask

    loop_weight = torch.full((num_nodes,), fillValue, dtype = None if edge_weight is None else edge_weight.dtype, device=edge_idx.device)
    if (edge_weight is not None):
        assert edge_weight.numel() == edge_idx.size(1)
        remaining_edge_weight = edge_weight[inv_mask]

        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)
    
    loop_idx = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_idx = loop_idx.unsqueeze(0).repeat(2, 1)
    edge_idx = torch.cat([edge_idx[:, mask], loop_idx], dim=1)

    return edge_idx, edge_weight

class _SGConv(SGConv):
    def __init__(self, num_features, num_classes, K=1, cached=False, bias=True):
        super(_SGConv, self).__init__(num_features, num_classes, K, cached, bias)
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
        fill_value = 1 if not improved else 2
        edge_idx, edge_weight = add_remaining_self_loops(edge_idx, edge_weight, fill_value, num_nodes)
        row, col = edge_idx
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_squrt = deg.pow(-0.5) # 度矩阵归一化
        deg_inv_squrt[deg_inv_squrt == float('inf')] = 0 # 规范除0
        # 归一化邻接矩阵
        return edge_idx, deg_inv_squrt[row] * edge_weight * deg_inv_squrt[col]
    
    def forward(self, x, edge_idx, edge_weight=None):

        # 缓存加速
        if not self.cached or self.cached_result is None:
            edge_idx, edge_weight = self.norm(edge_idx, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_idx, x=x, edge_weight=edge_weight)
            self.cached_result = x
        
        return self.lin(self.cached_result)
    
    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j
    

class DGCNN(nn.Module):
    def __init__(self, device, num_nodes, edge_weight, edge_idx, num_features, num_hiddens, num_classes, num_layers, learnable_edge_weight=True, dropout=0.5):
        """DGCNN model
        Args:
            device (torch.device): model device.
            num_nodes (int): number of nodes in the graph.
            edge_weight (tensor): edge matrix.
            edge_idx (tensor): edge index.
            num_features (int): number of features for each node/channel.
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

        # 使节点是可学习的 共 node_num 个节点，每个节点使用 num_features 个特征表示
        self.node_embedding = nn.Parameter(torch.randn(num_nodes, num_features).float(), requires_grad=True)
        nn.init.xavier_normal_(self.node_embedding)


        self.dropout = dropout
        self.bn1 = nn.BatchNorm1d(num_features)
        self.conv1 = _SGConv(num_features=num_features, num_classes=num_hiddens, K=num_layers)

        self.fc1 = nn.Linear(num_hiddens * num_nodes, 64)
        self.fc2 = nn.Linear(64, num_classes)

        # 已弃用
        self.conv2 = DeprecatedConvDescriptor(nn.Conv1d(self.num_nodes, 1, 1))
        self.fc = DeprecatedConvDescriptor(nn.Linear(num_hiddens, num_classes))
    
    def append(self, edge_idx: torch.Tensor, batch_size: int):
        """concate a batch of graphs.

        Args:
            edge_idx (edge tensor): edge idx of graph.
            batch_size (int): size of one batch.

        Returns:
            edge_idxAll: edge of concated
        """
        # edge_idx 的的形状： [[0,1,2,...], [0,1,2,...]]
        # 扩充后的形状： [[0,1,2,...,0,1,2,...], [0,1,2,...,0,1,2,...]]
        # 用于存储扩展后的边索引
        edge_idx = torch.LongTensor(edge_idx)
        edge_idx_all = torch.LongTensor(2, edge_idx.shape[1] * batch_size)
        # 用于存储batch索引
        data_batch = torch.LongTensor(self.num_nodes * batch_size)
        for i in range(batch_size):
            edge_idx_all[:, i*edge_idx.shape[1]:(i+1)*edge_idx.shape[1]] = edge_idx + i * self.num_nodes
            data_batch[i*self.num_nodes:(i+1)*self.num_nodes] = i
        return edge_idx_all.to(self.device), data_batch.to(self.device)

    def forward(self, x):
        batch_size = len(x)
        if self.node_embedding is not None:
            node_embedding = self.node_embedding.unsqueeze(0).repeat(batch_size, 1, 1)
            x = x + node_embedding
        edge_idx, _ = self.append(self.edge_idx, batch_size)
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_idx.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = normalize_matrix(edge_weight)
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(-1, x.shape[-1])
        x = self.conv1(x, edge_idx, edge_weight)
        x = x.view((batch_size, -1))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        return x
