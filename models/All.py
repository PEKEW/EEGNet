import torch
import torch.nn as nn
from models.CNNVAE import OutterBAE
from models.DGCNN import DGCNN
import torch.nn.functional as F

"""
DTF: Depersonalized Transfer Field

最终的包含所有模态的网络图
数据流：
原始的EEG -> DTF -> Depersionalized GNN (1
          -> DGCNN -> GNN （2

视频 -> CNNVAE-> GNN （3

CNNVAE  = (视频 -> BAE -> GNN)

BAE = (视频 -> VAE -> GNN
        光流 -> Attention -> GNN
         log -> Attention -> GNN)

损失：
对比学习损失 用于对比不同模态的GNN表示特征
去个性化损失 用于控制EEG中的个性化部分
分类损失
重建损失 用于更新BAE
未使用：表示一致性损失

可能需要的正则化项：
图结构正则化：稀疏图 用于减少噪声连接
图平滑正则化：让相邻节点的表示尽可能相似
表示正则化：让不同模态的表示尽可能相似！
去个性化正则化：让EEG中的个性化部分尽可能小
"""


class AttentionRes(nn.Module):
    def __init__(self, input_dim=930,
                 hidden_dim=256,
                 num_headers=8,):
        super(AttentionRes, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_headers = num_headers
        self.scale = (hidden_dim // num_headers) ** 0.5

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

        # res
        out = identity + out
        out = self.norm1(out)

        identity = out
        out = self.ffn(out)

        # res
        out = identity + out
        out = self.norm2(out)

        return out


class MCDIS(nn.Module):
    def __init__(self, args, edge_weight, edge_idx, DFT: torch.Module):
        super(MCDIS, self).__init__()

        self.args = args
        self.bae = OutterBAE(args, (33, 32))
        self.dgcn = DGCNN(args, edge_weight, edge_idx,
                          num_hiddens=args.num_hiddens,
                          num_layers=args.num_layers,
                          dropout=args.dropout,
                          node_learnable=args.node_learnable)
        self.DFT = DFT
        self.combine = AttentionRes(input_dim=930,
                                    hidden_dim=args.atn_hidden_dim,
                                    num_headers=args.num_headers)
        self.mlp_input_size = 930 (30*30 + 30)
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
        """ 对比损失 用来对齐两个模态的图表达的特征 """
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
        """ 去个性化损失 用于控制EEG中的个性化部分 通过mmd衡量两个图表达的分布距离
            使用中位数启发式方法计算高斯核的sigma
        """
        def rbf_kernal(x1, x2, sigma):
            distance = torch.cdist(x1, x2)
            return torch.exp(-distance / (2 * sigma ** 2))
        batch_size = personal_graph.shape[0]
        n_smaples = batch_size * 2
        total = torch.cat([personal_graph, depersonal_graph], dim=0)
        distances = torch.cdist(total, total)
        sigma = torch.median(distances) / \
            (2 * torch.log(torch.tensor(n_smaples + 1.0)))
        k_ss = rbf_kernal(personal_graph, personal_graph, sigma)
        k_tt = rbf_kernal(depersonal_graph, depersonal_graph, sigma)
        k_st = rbf_kernal(personal_graph, depersonal_graph, sigma)

        mmd = (k_ss.sum() / (batch_size * batch_size) + k_tt.sum() /
               (batch_size * batch_size) - 2 * k_st.sum() /
               (batch_size * batch_size))
        return mmd

    def _reconstruction_loss(self, video_graph, eeg_graph):
        """ 重建损失 用于更新BAE """
        mse_loss = F.mse_loss(video_graph, eeg_graph, reduction='mean')
        l1_loss = F.l1_loss(video_graph, eeg_graph, reduction='mean')
        total_recon_loss = mse_loss + l1_loss * 0.1
        return total_recon_loss

    def _class_loss(self, x, y):
        loss = F.cross_entropy(x, y)
        return loss

    def loss(self, personal_graph, video_graph,
             depersonal_graph, prediction, labels):
        w_nce_loss = self.args.alpha
        w_mmd_loss = self.args.beta
        w_reconstruction_loss = self.args.gamma
        w_class_loss = self.args.delta
        nce_loss = self._info_nce_loss(personal_graph, video_graph)
        mmd_loss = self._mmd_loss(personal_graph, depersonal_graph)
        reconstruction_loss = self._reconstruction_loss(
            video_graph, personal_graph)
        class_loss = self._class_loss(prediction, labels)
        total_loss = w_nce_loss * nce_loss + w_mmd_loss * mmd_loss + \
            w_reconstruction_loss * reconstruction_loss + \
            w_class_loss * class_loss
        return total_loss
