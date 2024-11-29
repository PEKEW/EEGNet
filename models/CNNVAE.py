import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.Config import Args
from models.Utils import *
import itertools


class CNNEncoder(nn.Module):
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)
        return x


class BAE(nn.Module):
    def __init__(self, input_dim, args):
        super(BAE, self).__init__()
        """
        input_dim → hidden_size → latent_dim → hidden_size → output_dim
        # 视觉: 枕叶（亮度 运动
            顶枕(空间感知 视觉引导运动)
            顶叶(空间位置)
            O PO P
        # 运动: 枕叶(运动 速度 加减速 连续性)
            颞叶(复杂运动 轨迹)
            顶叶(三维空间 旋转 运动轨迹)
            顶枕(三位运动 整合视觉运动信息 空间定位)
            O T P PO
            
            0   1   2  3  4  5  6  
        F   Fp1 Fp2 Fz F3 F4 F7 F8 
        
            7   8   9   10  
        Fc  Fc1 Fc2 Fc5 Fc6
        
            11 12 13 
        C   Cz C3 C4 
        
            14 15 
        T   T7 T8 
        
            16  17  18  19 
        Cp  Cp1 Cp2 Cp5 Cp6 
        
            20 21 22 23 24
        P   Pz P3 P4 P7 P8 
        
            25  26
        Po  Po3 Po4 
        
            27 28 29
        O   Oz O1 O2
        """
        node_dim = 30
        self.input_dim = input_dim
        self.hidden_size = args.hidden_size
        self.latent_dim = args.latent_size
        self.num_features = args.num_features
        self.edge_dim = node_dim * node_dim  # 30 * 30 = 900
        self.node_dim = node_dim
        self.output_size = self.edge_dim + node_dim * args.num_features
        dp = args.vae_dropout

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(dp)
        )

        self.fc_mu = nn.Linear(self.hidden_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_size, self.latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(dp),
            nn.Linear(self.hidden_size, self.output_size)
        )

        # P OP P ： idx = 20 21 22 23 24 25 26 27 28 29 len = 10
        # 10个节点互相连接，所以是100个边，但是连接对称加自环，所以只需要55个边
        self.optical_nodes = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        self.optical_edges = [
            (i, j) for i, j in itertools.product(self.optical_nodes, self.optical_nodes) if i <= j
        ]
        self.optical_node_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, len(self.optical_nodes)),
        )
        self.optical_edge_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, len(self.optical_edges)),
        )

        # P OP P T ： idx = 20 21 22 23 24 25 26 27 28 29 14 15 len = 12
        # 12个节点互相连接，所以是144个边，但是连接对称加自环，所以只需要78个边
        self.optical_temporal_nodes = [
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 14, 15]
        self.optical_temporal_edges = [
            (i, j) for i, j in itertools.product(self.optical_temporal_nodes, self.optical_temporal_nodes) if i <= j
        ]
        self.log_node_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, len(self.optical_temporal_nodes)),
        )
        # TODO: important log shape is not input_dim
        self.log_edge_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, len(self.optical_temporal_edges)),
        )

    def apply_attention(self, node_repr, edge_repr, node_weights, edge_weights):
        node_attention_weights = F.softmax(node_weights, dim=1)
        edge_attention_weights = F.softmax(edge_weights, dim=1)

        enhanced_node_repr = node_repr.clone()
        for idx, node in enumerate(self.optical_nodes):
            enhanced_node_repr[:, node, :] = node_repr[:, node, :] * \
                node_attention_weights[:, idx].unsqueeze(-1)

        enhanced_edge_repr = edge_repr.clone()
        for idx, edge in enumerate(self.optical_edges):
            enhanced_edge_repr[:, edge[0], edge[1]] = edge_repr[:, edge[0], edge[1]] * \
                edge_attention_weights[:, idx].unsqueeze(-1)

        return enhanced_node_repr, enhanced_edge_repr

    def encode(self, x):
        h1 = self.encoder(x)
        return self.fc_mu(h1), self.fc_logvar(h1)

    def decode(self, z):
        output = self.decoder(z)
        edge_repr = output[:, :self.edge_dim]
        node_repr = output[:, self.edge_dim:]
        edge_repr = edge_repr.view(-1, self.node_dim, self.node_dim)
        node_repr = node_repr.view(-1, self.node_dim, self.num_features)
        return edge_repr, node_repr

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, optical, log):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        edge_repr, node_repr = self.decode(z)

        optical_node_weights = self.optical_node_attention(optical)
        optical_edge_weights = self.optical_edge_attention(optical)
        node_repr, edge_repr = self.apply_attention(
            node_repr, edge_repr, optical_node_weights, optical_edge_weights)
        log_node_weights = self.log_node_attention(log)
        log_edge_weights = self.log_edge_attention(log)
        node_repr, edge_repr = self.apply_attention(
            node_repr, edge_repr, log_node_weights, log_edge_weights)

        return edge_repr, node_repr, mu, logvar


class OutterBAE(nn.Module):
    def __init__(self, input_size=(64, 64), node_dim=30, num_features=Args.num_features,
                 latent_dim=90, dropout_rate=0.5, weight_decay=1e-5):
        super(OutterBAE, self).__init__()
        self.args = Args()
        self.weight_decay = weight_decay
        self.node_dim = node_dim
        self.num_features = num_features
        self.cnn = CNNEncoder(self.args)
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            cnn_output = self.cnn(dummy_input)
            flattened_size = cnn_output.view(1, -1).size(1)
        self.bae = BAE(
            input_dim=flattened_size,
            args=self.args
        )

        combined_size = self.args.edge_hidden_size + self.args.node_hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),  # 增大全连接层维度
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2)

        )

        self.classifier_only_video = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2)
        )

        self.apply(self._init_weights)

    def log_processor(self, x):
        # TODO: important  这里具体怎么做需要看数据集的返回状态
        return x

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, original, optical, log):
        original = original.mean(dim=1)
        original = self.cnn(original).view(original.size(0), -1)

        optical = optical.mean(dim=1)
        optical = self.cnn(optical).view(optical.size(0), -1)

        log = self.log_processor(log)
        edge_repr, node_repr, *_ = self.bae(original, optical, log)
        return edge_repr, node_repr

    def forward_only_video(self, original, optical, log):
        x = original if original is not None else optical
        x = x.mean(dim=1)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(x.size(0), -1)
        output = self.classifier_only_video(cnn_features)
        return output


def get_cnn_model():
    return OutterBAE(
        input_size=(32, 32)
    )
