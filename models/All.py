import torch
import torch.nn as nn
from models.CNNVAE import OutterBAE
from models.DGCNN import DGCNN
import torch.nn.functional as F
from models.Utils import check_nan

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
        edge_weight = nn.Parameter(edge_weight, requires_grad=True)
        node_weight = nn.Parameter(node_weight, requires_grad=True)
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
            nn.Dropout(args.mcdis_mlp_dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(args.mcdis_mlp_dropout),
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
        # TODO: important personal 是pertrained edge depersonal 是输入的common 他们应该都是30 30的
        out = self.MLP(combined_feature)
        depersonal_graph = self._get_deperson_graph(eeg)
        if check_nan([out, personal_graph_gnn_out, video_graph_gnn_out, depersonal_graph]):
            raise ValueError('nan in forward')
        return out, personal_graph_gnn_out, video_graph_gnn_out, depersonal_graph

    def get_optimizer_contrastive(self):
        return torch.optim.Adam([
            {'params': self.pretrain_edge,
                'lr': self.args.contrastive_edge_lr},
            {'params': self.pretrain_node,
                'lr': self.args.contrastive_node_lr},
            {'params': self.gnn.parameters(), 'lr': self.args.contrastive_gnn_lr}]
        )

    def get_optimizer_rebuild(self):
        return torch.optim.Adam([
            {'params': self.bae.parameters(),
                'lr': self.args.rebuild_lr},]
        )

    def get_optimizer_all(self):
        # TODO: improve III lr controller, weight decay
        return torch.optim.Adam([
            {'params': self.parameters(),
                'lr': self.args.mcdis_lr},]
        )

    def _contrastive_loss(self, personal_graph, video_graph):
        # TODO: improve III gradient clipping
        """ contrastive loss NCE"""
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

    def _rebuild_loss(self, video_graph, eeg_graph):
        mse_loss = F.mse_loss(video_graph, eeg_graph, reduction='mean')
        l1_loss = F.l1_loss(video_graph, eeg_graph, reduction='mean')
        total_recon_loss = mse_loss + l1_loss * self.args.rebuild_l1_reg
        return total_recon_loss

    def _class_loss(self, x, y):
        loss = F.cross_entropy(x, y)
        return loss

    def loss(self, personal_graph, video_graph,
             depersonal_graph, prediction, labels, name='total'):
        # TODO: improve III warm-up strategy
        # TODO: improve III gradient noormalization
        w_contrast_loss = self.args.alpha
        w_reconstruction_loss = self.args.gamma
        w_class_loss = self.args.delta
        contrast_loss = self._contrastive_loss(personal_graph, video_graph)
        rebuild_loss = self._rebuild_loss(
            video_graph, personal_graph)
        class_loss = self._class_loss(prediction, labels)
        w_struct = self.args.struct_reg
        # struct_reg = self._depersonal_reg(personal_graph, depersonal_graph)
        total_loss = w_contrast_loss * contrast_loss +  \
            w_reconstruction_loss * rebuild_loss + \
            w_class_loss * class_loss + \
            0
            # w_struct * struct_reg
        if check_nan([total_loss, contrast_loss, class_loss, rebuild_loss]):
            raise ValueError('nan in loss')
        return {
            'total_loss': total_loss,
            'contrast_loss': contrast_loss,
            'class_loss': class_loss,
            'rebuild_loss': rebuild_loss,
        }

    def _structure_reg(self, pretrain_edge, depersonal_graph):
        """
        binary cross entropy
        """
        edge_binary = (pretrain_edge > 0).float()
        deper_binary = (depersonal_graph > 0).float()
        structure_diff = F.binary_cross_entropy(
            edge_binary, 
            deper_binary,
            reduction='mean'
        )
        return structure_diff

    def _mmg_reg(self, personal_graph, depersonal_graph):
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

    def _depersonal_reg(self, personal_graph, depersonal_graph):
        mmd = self._mmg_reg(personal_graph, depersonal_graph)
        structure_diff = self._structure_reg(
            self.pretrain_edge, depersonal_graph)
        mmd_w = self.args.mmd_w
        struce_w = self.args.struce_w
        return  mmd_w * mmd + struce_w * structure_diff
    