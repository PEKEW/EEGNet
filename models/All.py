import torch
import torch.nn as nn
from models.CNNVAE import OutterBAE
from models.DGCNN import DGCNN
import torch.nn.functional as F
from models.Dtf import DTF

"""
DTF: Depersonalized Transfer Field

final graph include all modal
data flow:
original EEG -> DTF -> Depersonalized GNN (1
            -> DGCNN -> GNN (2
video -> CNNVAE -> GNN (3
CNNVAE = (video -> BAE -> GNN)
BAE = (video -> VAE -> GNN
            optical flow -> Attention -> GNN
            log -> Attention -> GNN)
"""


class MCDIS(nn.Module):
    def __init__(self, args, edge_weight, edge_idx, node_weight, dtf):
        super(MCDIS, self).__init__()

        self.args = args
        # INFO: BAE:
        # input video, optical , motion | output edge expr, node expr
        self.bae = OutterBAE((32, 32))
        self.pretrain_edge = edge_weight
        self.pretrain_node = node_weight
        self.edge_idx = edge_idx
        self.gnn = DGCNN(edge_weight, edge_idx,
                         num_hiddens=args.gnn_hiddens_size,
                         num_layers=args.gnn_num_layers,
                         dropout=args.gnn_dropout,
                         node_learnable=args.node_learnable)
        self.dtf = dtf
        self.mlp_input_size = 128  # two 64-dim feature combined
        # TODO: improve make hard code to args
        self.MLP = nn.Sequential(
            nn.Linear(self.mlp_input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(args.dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(args.dropout),
            nn.Linear(32, 2)
        )

    def _get_deperson_graph(self, eeg):
        return self.dtf(eeg)

    def forward(self, eeg, video, flow, log):
        # video graph: ((e), (n)), e: 16, 30, 30; n: 16, 30, 250
        video_graph = self.bae(video, flow, log)
        edge_expr, node_expr = video_graph
        video_graph_gnn_out = self.gnn.forward_embedding_with_batch(
            x=eeg,
            edge_weight=edge_expr,
            node_embedding=node_expr)
        # personal graph by gnn: 16, 30, 64(hidden)
        personal_graph_gnn_out = self.gnn.forward_embedding_with_batch(
            x=eeg,
            edge_weight=self.pretrain_edge,
            node_embedding=self.pretrain_node)
        combined_feature = torch.cat(
            [personal_graph_gnn_out, video_graph_gnn_out], dim=2)
        combined_feature, _ = torch.max(combined_feature, dim=1)

        out = self.MLP(combined_feature)
        depersonal_graph = self._get_deperson_graph(eeg)
        return out, personal_graph_gnn_out, video_graph_gnn_out, depersonal_graph

    def get_optimizer_nce(self):
        return torch.optim.Adam([
            {'pretrain_edge': self.pretrain_edge, 'lr': self.args.nce_edge_lr},
            {'pretrain_node': self.pretrain_node, 'lr': self.args.nce_node_lr},
            {'gnn': self.gnn.parameters(), 'lr': self.args.nce_gnn_lr}]
        )

    def _info_nce_loss(self, personal_graph, video_graph):
        """ contrastive loss """
        temp = self.args.temperature
        eeg = F.normalize(personal_graph, dim=1)
        video = F.normalize(video_graph, dim=1)
        eeg = eeg.mean(dim=1)
        video = video.mean(dim=1)
        
        logits = torch.matmul(eeg, video.t()) / temp
        batch_size = eeg.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        loss = F.cross_entropy(logits, labels) + \
            F.cross_entropy(logits.t(), labels)
        return loss / 2

    def _mmd_loss(self, personal_graph, depersonal_graph):
        # TODO: improve mmd_loss should be renamed as mmd_reg
        def rbf_kernal(x1, x2, sigma):
            distance = torch.cdist(x1, x2)
            return torch.exp(-distance / (2 * sigma ** 2))
        batch_size = personal_graph.shape[0]
        n_samples = batch_size * 2
        total = torch.cat([personal_graph, depersonal_graph], dim=0)
        distances = torch.cdist(total, total)
        sigma = torch.median(distances) / \
            (2 * torch.log(torch.tensor(n_samples + 1.0)))
        k_ss = rbf_kernal(personal_graph, personal_graph, sigma)
        k_tt = rbf_kernal(depersonal_graph, depersonal_graph, sigma)
        k_st = rbf_kernal(personal_graph, depersonal_graph, sigma)

        mmd = (k_ss.sum() / (batch_size * batch_size) + k_tt.sum() /
               (batch_size * batch_size) - 2 * k_st.sum() /
               (batch_size * batch_size))
        return mmd

    def _reconstruction_loss(self, video_graph, eeg_graph):
        mse_loss = F.mse_loss(video_graph, eeg_graph, reduction='mean')
        l1_loss = F.l1_loss(video_graph, eeg_graph, reduction='mean')
        total_recon_loss = mse_loss + l1_loss * 0.1
        return total_recon_loss

    def _class_loss(self, x, y):
        loss = F.cross_entropy(x, y)
        return loss

    def loss(self, personal_graph, video_graph,
        depersonal_graph, prediction, labels, name='total'):
        # TODO: important where should cal the mmd reg?
        w_nce_loss = self.args.alpha
        w_reconstruction_loss = self.args.gamma
        w_class_loss = self.args.delta
        nce_loss = self._info_nce_loss(personal_graph, video_graph)
        reconstruction_loss = self._reconstruction_loss(
            video_graph, personal_graph)
        class_loss = self._class_loss(prediction, labels)
        total_loss = w_nce_loss * nce_loss +  \
            w_reconstruction_loss * reconstruction_loss + \
            w_class_loss * class_loss
        return {
            'total_loss': total_loss,
            'contrast_loss': nce_loss,
            'cross_loss': class_loss,
            'rebuild_loss': reconstruction_loss,
        }
