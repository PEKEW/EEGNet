import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.Config import Args
from models.Utils import *
import itertools
class CNNEncoder(nn.Module):
    # def __init__(self, args):
    #     super(CNNEncoder, self).__init__()
    #     self.conv1 = nn.Conv2d(3, args.channels1, kernel_size=3, padding=1)
    #     self.conv2 = nn.Conv2d(16, args.channels2, kernel_size=3, padding=1)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.batch_norm1 = nn.BatchNorm2d(args.channels1)
    #     self.batch_norm2 = nn.BatchNorm2d(args.channels2)
        
    # def forward(self, x):
    #     x = F.relu(self.batch_norm1(self.conv1(x)))
    #     x = self.pool(x)
    #     x = F.relu(self.batch_norm2(self.conv2(x)))
    #     x = self.pool(x)
    #     return x
    
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        # 增加卷积层深度和通道数
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 增加通道数到64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 增加到128
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 新增一层
        
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
    def __init__(self, input_dim, args,):
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
        self.edge_dim = node_dim * node_dim # 30 * 30 = 900
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
            (i,j) for i, j in itertools.product(self.optical_nodes, self.optical_nodes) if i <= j
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
        self.optical_temporal_nodes = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 14, 15]
        self.optical_temporal_edges = [
            (i,j) for i, j in itertools.product(self.optical_temporal_nodes, self.optical_temporal_nodes) if i <= j
        ]
        self.log_node_attention = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, len(self.optical_temporal_nodes)),
        )
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
        node_repr, edge_repr = self.apply_attention(node_repr, edge_repr, optical_node_weights, optical_edge_weights)
        log_node_weights = self.log_node_attention(log)
        log_edge_weights = self.log_edge_attention(log)
        node_repr, edge_repr = self.apply_attention(node_repr, edge_repr, log_node_weights, log_edge_weights)
        
        return edge_repr, node_repr, mu, logvar
    

class CNNVAE(nn.Module):
    def __init__(self, input_size=(64, 64), node_dim=30, num_features=Args.num_features, 
                latent_dim=90, dropout_rate=0.5, weight_decay=1e-5):
        super(CNNVAE, self).__init__()
        self.args = Args()
        self.weight_decay = weight_decay
        self.node_dim = node_dim
        self.num_features = num_features
        self.cnn = CNNEncoder(self.args)
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            cnn_output = self.cnn(dummy_input)
            flattened_size = cnn_output.view(1, -1).size(1)
        self.vae = VAE(
            input_dim=flattened_size,
            latent_dim=latent_dim,
            node_dim=node_dim,
            num_features=num_features,
            args = self.args
        )
        self.edge_processor = nn.Sequential(
            nn.Linear(node_dim * node_dim, self.args.edge_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.edge_hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.node_processor = nn.Sequential(
            nn.Linear(num_features, self.args.node_hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(self.args.node_hidden_size),
            nn.Dropout(dropout_rate)
        )
        
        combined_size = self.args.edge_hidden_size + self.args.node_hidden_size
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
        
        
        self.classifier_only_video = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )
        
        
        self.classifier_only_video = nn.Sequential(
            nn.Linear(4096, 512),  # 增大全连接层维度
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
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def process_edge_features(self, edge_repr):
        batch_size = edge_repr.size(0)
        return edge_repr.reshape(batch_size, -1)
        # graph_features = self.edge_processor(edge_repr.reshape(32,-1))
        # return graph_features.view(edge_repr.size(0), -1)
    
    def process_node_features(self, node_repr):
        batch_size = node_repr.size(0)
        return node_repr.reshape(batch_size, -1)
        # node_features = self.node_processor(node_repr.reshape(batch_size, -1))
        # return node_features.view(batch_size, -1)
    
    
    
    # mark only video forward edition
    def forward(self, original, optical):
        x = original if original is not None else optical
        x = x.mean(dim=1)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(x.size(0), -1)
        output = self.classifier_only_video(cnn_features)
        return output
    
    # mark cnn vae forward edition
    # def forward(self, original, optical):
    #     # todo 把32这种硬编码的数字改成参数 放在config里面
    #     # todo 这里需要有一个batch size 和 device参数
    #     x1 = original if original is not None else torch.zeros(32, 29, 3, 32, 32)
    #     x2 = optical if optical is not None else torch.zeros(32, 29, 3, 32, 32)
    #     # todo 这里在最终调整网络模型的时候需要改成attention机制
    #     x = x1.to(torch.device('cuda')) + x2.to(torch.device('cuda'))
    #     x = x.mean(dim=1)
    #     cnn_features = self.cnn(x)
    #     cnn_features = cnn_features.view(x.size(0), -1)
    #     edge_repr, node_repr, *_= self.vae(cnn_features)
    #     node_features = self.process_node_features(node_repr)
    #     edge_features = self.process_edge_features(edge_repr)
    #     combined_features = torch.cat([edge_features, node_features], dim=1)
    #     output = self.classifier(combined_features)
    #     return output, edge_repr
    
    
    def get_l1_l2_regularization(self):
        """计算L1和L2正则化损失"""
        l1_loss = 0
        l2_loss = 0
        
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
            l2_loss += torch.square(param).sum()
            
        return l1_loss, l2_loss

    def vae_loss(self, recon_edge, recon_node, orig_features, mu, logvar, beta=1.0):
        """计算VAE损失"""
        edge_loss = F.mse_loss(recon_edge.view(recon_edge.size(0), -1), 
                              orig_features[:, :self.node_dim * self.node_dim], 
                            reduction='mean')
        
        node_loss = F.mse_loss(recon_node.view(recon_node.size(0), -1),
                              orig_features[:, self.node_dim * self.node_dim:],
                            reduction='mean')
        
        # KL散度
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return edge_loss + node_loss + beta * kld

def train_step(model, x, labels, optimizer, beta=1.0):
    optimizer.zero_grad()
    predictions, edge_repr, node_repr, mu, logvar = model(x)
    classification_loss = F.cross_entropy(predictions, labels)
    cnn_features = model.cnn(x).view(x.size(0), -1)
    vae_loss = model.vae_loss(edge_repr, node_repr, cnn_features, mu, logvar, beta=beta)
    l1_loss, l2_loss = model.get_l1_l2_regularization()
    total_loss = (classification_loss + 
                  0.1 * vae_loss + 
                  model.weight_decay * (0.1 * l1_loss + l2_loss))
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {
        'total_loss': total_loss.item(),
        'classification_loss': classification_loss.item(),
        'vae_loss': vae_loss.item(),
        'l1_loss': l1_loss.item(),
        'l2_loss': l2_loss.item()
    }
    
    
def get_cnn_model():
    return CNNVAE(
        input_size=(32,32)
    )