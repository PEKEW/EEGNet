from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_scatter import scatter_add
from models.Utils import normalize_matrix, normalize_matrix_batch


def maybe_num_nodes(edge_idx, num_nodes=None):
    return edge_idx.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_idx, edge_weight, fill_value=1, num_nodes=None):
    """
    add self-loops
    """
    num_nodes = maybe_num_nodes(edge_idx, num_nodes)
    row, col = edge_idx
    mask = row != col
    inv_mask = ~mask

    loop_weight = torch.full(
        (num_nodes,), fill_value, dtype=None if edge_weight is None else edge_weight.dtype, device=edge_idx.device)
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
    def __init__(self, num_features, num_classes, K=2, cached=False, bias=True):
        super(_SGConv, self).__init__(
            num_features, num_classes, K, cached, bias)
        self.cached_result = None
        nn.init.xavier_normal_(self.lin.weight)
        if K < 1:
            raise ValueError(f"K should be >= 1, got {K}")

    @staticmethod
    def norm(edge_idx, num_nodes, edge_weight, improved=False, dtype=None):
        """Normalization of the graph adjacency matrix.
        """
        if edge_weight is None:
            edge_weight = torch.ones((edge_idx.size(1), ),
                                     dtype=dtype,
                                     device=edge_idx.device)

        fill_value = 2 if improved else 1
        edge_idx, edge_weight = add_remaining_self_loops(
            edge_idx, edge_weight, fill_value, num_nodes)

        row, col = edge_idx
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_idx, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.cached or self.cached_result is None:
            edge_index, norm_weight = self.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            self.cached_result = x
            for k in range(self.K):
                self.cached_result = self.propagate(edge_index,
                                                    x=self.cached_result,
                                                    edge_weight=norm_weight)

        return self.lin(self.cached_result)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class DGCNN(nn.Module):
    def __init__(self,
                 edge_weight,
                 edge_idx,
                 num_features=250,
                 num_nodes=30,
                 num_hiddens=64,
                 num_classes=2,
                 num_layers=2,
                 dropout=0.5,
                 node_learnable=False):
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
            dropout (float, optional): dropout. Defaults to 0.5.
        """
        super(DGCNN, self).__init__()
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(
            self.num_nodes, self.num_nodes, offset=0)
        self.edge_idx = edge_idx
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[
            self.xs, self.ys]
        self.edge_weight = nn.Parameter(torch.Tensor(
            edge_weight).float(), requires_grad=True)

        if node_learnable:
            self.node_embedding = nn.Parameter(torch.randn(
                num_nodes, num_features).float(), requires_grad=True)
            nn.init.xavier_normal_(self.node_embedding)
        else:
            self.node_embedding = None
        self.node_learnable = node_learnable

        self.dropout = dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.bn1 = nn.BatchNorm1d(num_features)
        self.conv1 = _SGConv(num_features=num_features,
                             num_classes=num_hiddens, K=num_layers)
        fc_input_dim = num_hiddens * num_nodes
        self.hidden_sizes = [
            fc_input_dim,
            fc_input_dim // 2,
            fc_input_dim // 4,
            fc_input_dim // 8,
            num_classes
        ]

        self.fc_layers = nn.ModuleList()
        for i in range(len(self.hidden_sizes) - 1):
            self.fc_layers.append(
                nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))

        for fc in self.fc_layers:
            nn.init.xavier_normal_(fc.weight)
            if fc.bias is not None:
                nn.init.zeros_(fc.bias)

    def append(self, edge_idx, batch_size):
        """Concatenate a batch of graphs.
        Args:
            edge_idx (torch.Tensor): Edge indices of the graph. Expected shape: [2, E] where E is the number of edges.
            batch_size (int): Number of graphs to concatenate.
        Returns:
            tuple: A tuple containing:
                - edge_idx_all (torch.Tensor): Concatenated edge indices
                - data_batch (torch.Tensor): Batch assignment for each node
        """
        if not isinstance(edge_idx, torch.Tensor):
            edge_idx = torch.LongTensor(edge_idx)

        edge_idx_all = torch.zeros((2, edge_idx.shape[1] * batch_size),
                                   dtype=torch.long)
        data_batch = torch.zeros(self.num_nodes * batch_size,
                                 dtype=torch.long)
        for i in range(batch_size):
            start_idx = i * edge_idx.shape[1]
            end_idx = (i + 1) * edge_idx.shape[1]
            edge_idx_all[:, start_idx:end_idx] = edge_idx + i * self.num_nodes
            start_node = i * self.num_nodes
            end_node = (i + 1) * self.num_nodes
            data_batch[start_node:end_node] = i
        device = next(self.parameters()).device
        return edge_idx_all.to(device), data_batch.to(device)

    def forward(self, x):
        batch_size = len(x)
        if self.node_embedding is not None:
            node_embedding = self.node_embedding.unsqueeze(
                0).repeat(batch_size, 1, 1)
            x = x + node_embedding
        edge_idx, _ = self.append(self.edge_idx, batch_size)
        edge_weight = torch.zeros(
            (self.num_nodes, self.num_nodes), device=edge_idx.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(
            edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + \
            edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = normalize_matrix(edge_weight)
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(-1, x.shape[-1])
        x = self.conv1(x, edge_idx, edge_weight)
        x = x.view((batch_size, -1))

        x = self.dropout_layer(x)

        for i, fc_layer in enumerate(self.fc_layers):
            x = fc_layer(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)

        return x

    def forward_embedding(self, x):
        batch_size = len(x)
        if self.node_embedding is not None:
            node_embedding = self.node_embedding.unsqueeze(
                0).repeat(batch_size, 1, 1)
            x = x + node_embedding
        edge_idx, _ = self.append(self.edge_idx, batch_size)
        edge_weight = torch.zeros(
            (self.num_nodes, self.num_nodes), device=edge_idx.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(
            edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + \
            edge_weight.transpose(1, 0) - torch.diag(edge_weight.diagonal())
        edge_weight = normalize_matrix(edge_weight)
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(-1, x.shape[-1])
        x = self.conv1(x, edge_idx, edge_weight)
        x = x.view((batch_size, -1))
        # [16, 30, 64]
        x = x.reshape(batch_size, self.num_nodes, -1)
        return x

    def forward_embedding_with_batch(self, x, edge_weight, node_embedding):
        x = x + node_embedding
        batch_size = len(x)
        if edge_weight.dim() == 2:
            edge_weight = edge_weight.unsqueeze(0).repeat(16, 1, 1)
        edge_idx, _ = self.append(self.edge_idx, batch_size)
        edge_weight = normalize_matrix_batch(edge_weight)
        edge_weight = edge_weight.reshape(-1)
        x = self.bn1(x.transpose(1, 2)).transpose(1, 2)
        x = x.reshape(-1, x.shape[-1])
        x = self.conv1(x, edge_idx, edge_weight)
        x = x.view((batch_size, -1))
        x = x.reshape(batch_size, self.num_nodes, -1)
        return x


def get_gnn_model(args, edge_wight, edge_idx):
    return DGCNN(
        num_nodes=args.num_nodes,
        edge_weight=edge_wight,
        edge_idx=edge_idx,
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_hiddens=args.num_hiddens,
        num_layers=args.num_layers,
    )
