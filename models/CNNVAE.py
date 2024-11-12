import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.Config import Args

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.batch_norm2 = nn.BatchNorm2d(32)
        
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=90, node_dim=30, num_features=None):
        super(VAE, self).__init__()
        if num_features is None:
            raise ValueError("num_features must be specified")
            
        self.node_dim = node_dim
        self.num_features = num_features
        self.edge_dim = node_dim * node_dim  # 30 * 30 = 900
        self.output_dim = self.edge_dim + node_dim * num_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3)
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, self.output_dim)
        )
        
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
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        edge_repr, node_repr = self.decode(z)
        return edge_repr, node_repr, mu, logvar

class CompleteModel(nn.Module):
    def __init__(self, input_size=(64, 64), node_dim=30, num_features=Args.num_features, 
                 latent_dim=90, dropout_rate=0.5, weight_decay=1e-5):
        super(CompleteModel, self).__init__()
        self.weight_decay = weight_decay
        self.node_dim = node_dim
        self.num_features = num_features
        self.cnn = CNNEncoder(dropout_rate=dropout_rate)
        dummy_input = torch.zeros(1, 3, input_size[0], input_size[1])
        with torch.no_grad():
            cnn_output = self.cnn(dummy_input)
            flattened_size = cnn_output.view(1, -1).size(1)
        self.vae = VAE(
            input_dim=flattened_size,
            latent_dim=latent_dim,
            node_dim=node_dim,
            num_features=num_features
        )
        self.graph_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2)
        )
        self.node_processor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate)
        )
        
        graph_feature_size = 16 * (node_dim//2) * (node_dim//2) 
        node_feature_size = 32 * node_dim  
        combined_size = graph_feature_size + node_feature_size + latent_dim
        
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
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def process_graph_features(self, edge_repr):
        edge_repr = edge_repr.unsqueeze(1)  # [batch, 1, node_dim, node_dim]
        graph_features = self.graph_conv(edge_repr)
        return graph_features.view(edge_repr.size(0), -1)
    
    def process_node_features(self, node_repr):
        batch_size = node_repr.size(0)
        node_features = self.node_processor(node_repr.view(-1, self.num_features))
        return node_features.view(batch_size, -1)
    
    def forward(self, x):
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(x.size(0), -1)
        edge_repr, node_repr, mu, logvar = self.vae(cnn_features)
        graph_features = self.process_graph_features(edge_repr)
        node_features = self.process_node_features(node_repr)
        combined_features = torch.cat([graph_features, node_features, mu], dim=1)
        output = self.classifier(combined_features)
        
        return output, edge_repr, node_repr, mu, logvar
    
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