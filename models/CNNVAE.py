import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.Config import Args
from models.Utils import *
import itertools


class BaseAttention(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (in_dim + out_dim) // 2
        self.q = nn.Linear(in_dim, hidden_dim)
        self.k = nn.Linear(in_dim, hidden_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.scale = hidden_dim ** 0.5
        # TODO: important move to args
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        q = self.q(x).unsqueeze(1)
        k = self.k(x).unsqueeze(1)
        v = self.v(x).unsqueeze(1)
        att_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        att_weight = F.softmax(att_scores, dim=-1)
        att_weight = self.dropout(att_weight)
        out = torch.matmul(att_weight, v)
        return out.squeeze(1)


class OpticalNodeAttention(BaseAttention):
    def __init__(self, in_dim=1024, out_dim=10):
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden_dim=512)


class OpticalEdgeAttention(BaseAttention):
    def __init__(self, in_dim=1024, out_dim=55):
        super().__init__(in_dim=in_dim, out_dim=out_dim, hidden_dim=512)


class LogNodeAttention(BaseAttention):
    def __init__(self, in_dim=9, out_dim=12):
        super().__init__(in_dim=in_dim, out_dim=out_dim)


class LogEdgeAttention(BaseAttention):
    def __init__(self, in_dim=9, out_dim=55):
        super().__init__(in_dim=in_dim, out_dim=out_dim)


class CNNEncoder(nn.Module):
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, args.channels1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            args.channels1, args.channels2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            args.channels2, args.channels3, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.batch_norm1 = nn.BatchNorm2d(args.channels1)
        self.batch_norm2 = nn.BatchNorm2d(args.channels2)
        self.batch_norm3 = nn.BatchNorm2d(args.channels3)
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
        # visual : O P PO P
        ## light, move, space, location
        # move: O T P PO
        ## move speed acceleration continuity space trajectory  rotation
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
        # time, speed, acceleration, rotation_speed, is_sickness, complete_sickness, pos[0], pos[1], pos[2]
        self.log_dim = 9
        self.hidden_size = args.bae_hidden_size
        self.latent_dim = args.bae_latent_size
        self.num_features = args.num_features
        self.edge_dim = node_dim * node_dim  # 30 * 30 = 900
        self.node_dim = node_dim
        self.output_size = self.edge_dim + node_dim * args.num_features
        dp = args.bae_dropout
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
        # 10 node connection with each other, so 100 edges, but symmetric connection plus self-loop, so only 55 edges
        self.optical_nodes = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
        self.optical_edges = [
            (i, j) for i, j in itertools.product(self.optical_nodes, self.optical_nodes) if i <= j
        ]
        # P OP P T ： idx = 20 21 22 23 24 25 26 27 28 29 14 15 len = 12
        # same with the above
        self.log_nodes = [
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 14, 15]
        self.log_edges = [
            (i, j) for i, j in itertools.product(self.log_nodes, self.log_nodes) if i <= j
        ]
        self.optical_node_attention = OpticalNodeAttention(
            in_dim=self.input_dim, out_dim=len(self.optical_nodes))
        self.optical_edge_attention = OpticalEdgeAttention(
            in_dim=self.input_dim, out_dim=len(self.optical_edges))
        self.log_node_attention = LogNodeAttention(
            in_dim=self.log_dim, out_dim=len(self.log_nodes))
        self.log_edge_attention = LogEdgeAttention(
            in_dim=self.log_dim, out_dim=len(self.log_edges))

    # TODO: important +  or *？
    def apply_attention(self, node_repr, edge_repr, attended_node, attended_edge, edge_list, node_list):
        enhanced_node_repr = node_repr.clone()
        for idx, node in enumerate(node_list):
            enhanced_node_repr[:, node, :] = \
                (node_repr[:, node, :]) + \
                (attended_node[:, idx].unsqueeze(-1))

        enhanced_edge_repr = edge_repr.clone()
        for idx, edge in enumerate(edge_list):
            enhanced_edge_repr[:, edge[0], edge[1]] = \
                ((edge_repr[:, edge[0], edge[1]].view(-1, 1)) +
                    (attended_edge[:, idx].unsqueeze(1))).view(-1)

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
        edge_repr = (edge_repr + edge_repr.transpose(-2, -1)) / 2
        attended_optical_node = self.optical_node_attention(optical)
        attended_optical_edge = self.optical_edge_attention(optical)
        node_repr, edge_repr = self.apply_attention(
            node_repr, edge_repr, attended_optical_node, attended_optical_edge, self.optical_edges, self.optical_nodes)
        # TODO: important need normalization?
        log_mean = log.mean(1)
        attended_log_node = self.log_node_attention(log_mean)
        attended_log_edge = self.log_edge_attention(log_mean)
        node_repr, edge_repr = self.apply_attention(
            node_repr, edge_repr, attended_log_node, attended_log_edge, self.log_edges, self.log_nodes)
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

        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 512),
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
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 2)
        )

        self.apply(self._init_weights)

    def log_processor(self, x):
        # TODO: important
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


def get_cnn_model():
    return OutterBAE(
        input_size=(32, 32)
    )
