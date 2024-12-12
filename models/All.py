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


class AttentionRes(nn.Module):
    def __init__(self, input_dim=930,
                hidden_dim=256,
                num_heads=8,):
        super(AttentionRes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scale = (hidden_dim // num_heads) ** 0.5

        self.eeg_proj = nn.Linear(input_dim, hidden_dim)
        self.video_proj = nn.Linear(input_dim, hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, eeg, video):
        # ERROR: eeg shape " 16, 2" 
        batch_size = eeg.shape[0]

        eeg_hidden = self.eeg_proj(eeg)
        video_hidden = self.video_proj(video)

        identity = eeg

        q = self.q_proj(eeg_hidden).reshape(
            batch_size, -1, self.num_headers,
            self.hidden_dim // self.num_headers).transpose(1, 2)
        k = self.k_proj(video_hidden).reshape(
            batch_size, -1, self.num_headers,
            self.hidden_dim // self.num_headers).transpose(1, 2)
        v = self.v_proj(video_hidden).reshape(
            batch_size, -1, self.num_headers,
            self.hidden_dim // self.num_headers).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, -1, self.hidden_dim)
        out = self.out_proj(out)
        out = identity + out
        out = self.norm1(out)
        identity = out
        out = self.ffn(out)
        out = identity + out
        out = self.norm2(out)

        return out


class MCDIS(nn.Module):
    def __init__(self, args, edge_weight, edge_idx, Dtf):
        super(MCDIS, self).__init__()

        self.args = args
        self.bae = OutterBAE((32, 32))
        self.dgcn = DGCNN(edge_weight, edge_idx,
                        num_hiddens=args.gnn_hiddens_size,
                        num_layers=args.gnn_num_layers,
                        dropout=args.gnn_dropout,
                        node_learnable=args.node_learnable)
        self.Dtf = DTF
        self.combine = AttentionRes(input_dim=930,
                                    hidden_dim=args.atn_hidden_dim,
                                    num_heads=args.num_heads)
        self.mlp_input_size = 930 # (30*30 + 30)
        self.MLP = nn.Sequential(
            nn.Linear(self.mlp_input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(args.dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(args.dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(args.dropout),
            nn.Linear(32, 2)
        )

    def _get_deperson_graph(self, eeg):
        return self.DFT(eeg)

    def forward(self, eeg, video, flow, log):
        personal_graph = self.dgcn(eeg)
        video_graph = self.bae(video, flow, log)
        combined_feature = self.combine(personal_graph, video_graph)
        out = self.MLP(combined_feature)
        depersonal_graph = self._get_deperson_graph(eeg)
        return out, personal_graph, video_graph, depersonal_graph

    def _info_nce_loss(self, personal_graph, video_graph):
        """ contrastive loss """
        temp = self.args.temperature
        eeg = F.normalize(personal_graph, dim=1)
        video = F.normalize(video_graph, dim=1)
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
            'constrast_loss': nce_loss,
            'cross_loss': class_loss,
            'rebuild_loss': reconstruction_loss,
        }
