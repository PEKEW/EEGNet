import torch
import numpy as np
import math
from typing import List, Tuple
from itertools import product
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.path import Path
import matplotlib.patches as patches
from sklearn.preprocessing import normalize

total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 Fc1 Fc2 Fc5 Fc6 Cz C3 C4 T7 T8 Cp1 Cp2 Cp5 Cp6 Pz P3 P4 P7 P8 Po3 Po4 Oz O1 O2'''.split()
regions_mapping = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],  # 前额叶
    'Frontocentral': ['Fc1', 'Fc2', 'Fc5', 'Fc6'],  # 额中央区
    'Central': ['C3', 'C4', 'Cz'],  # 中央区
    'Temporal': ['T7', 'T8'],  # 颞叶
    'Centroparietal': ['Cp1', 'Cp2', 'Cp5', 'Cp6'],  # 中央顶区
    'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],  # 顶叶
    'Parietooccipital': ['Po3', 'Po4'],  # 顶枕区
    'Occipital': ['O1', 'O2', 'Oz']  # 枕叶
}
colors = {
        'Frontal': '#4299e1', 
        'Frontocentral': '#f56565', 
        'Central': '#9f7aea',    
        'Temporal': '#48bb78',   
        'Centroparietal': '#ed64a6',
        'Parietal': '#ecc94b', 
        'Parietooccipital': '#667eea', 
        'Occipital': '#ed8936'
    }

class DeidentificationInput:
    """输入数据处理类"""
    def __init__(self):
        self.module_indices = [
            list(range(0, 7)),     # 前额叶
            list(range(7, 11)),    # 额中央区
            list(range(11, 14)),   # 中央区
            list(range(14, 16)),   # 颞叶
            list(range(16, 20)),   # 中央顶区
            list(range(20, 25)),   # 顶叶
            list(range(25, 27)),   # 顶枕区
            list(range(27, 30))    # 枕叶
        ]


class CommonExtraction:
    """
    min ||G1 - W * H1||^2 + ||G2 - W * H2||^2
    s.t. W >= 0, H1 >= 0, H2 >= 0
    numerator = G1@H1.T + G2@H2.T
    denominator = (H1@H1.T + H2@H2.T ) @ W
    W := W * numerator / (denominator + eps)
    现在的思路基于NMF的思想，使用乘法法则和迭代求和G1 G2相同形状的基底W和分别的激活矩阵H1和H2
    基本思想：基底需要一致 但是允许每个图有自己的激活模型
    共同特征一定会在H1和H2中表现出相似的激活模式
    最后使用JS散度求H1和H2的强共同激活模式
    不能对H1和H2进行相似约束的原因是：可能会强制把G1独特模式mapping到H2上
    反之同理，这会削弱每个激活模式的独特性
    """
    def __init__(self, G1, G2, max_iter=500):
        self.eps = 1e-10
        self.G1 = G1
        self.G2 = G2
        _shape = G1.shape
        W = np.random.rand(_shape[0], _shape[1])
        H1 = np.random.rand(_shape[0], _shape[1])
        H2 = np.random.rand(_shape[0], _shape[1])
        self.max_iter = max_iter
        
        self.W = normalize(W, axis=0)
        self.H1 = normalize(H1, axis=0)
        self.H2 = normalize(H2, axis=0)
        
        self.patten_analysis = []
    
    def _get_common_base(self):
        for _ in range(self.max_iter):
            W_old = self.W.copy()
            H1_old = self.H1.copy()
            H2_old = self.H2.copy()
            
            numerator = np.dot(self.G1, self.H1.T) + np.dot(self.G2, self.H2.T)
            denominator = np.dot(self.W, np.dot(self.H1, self.H1.T) + np.dot(self.H2, self.H2.T))
            self.W *= numerator / (denominator + self.eps)
            
            numerator = np.dot(self.W.T, self.G1)
            denominator = np.dot(np.dot(self.W.T, self.W), self.H1)
            self.H1 *= numerator / (denominator + self.eps)
            
            numerator = np.dot(self.W.T, self.G2)
            denominator = np.dot(np.dot(self.W.T, self.W), self.H2)
            self.H2 *= numerator / (denominator + self.eps)
            
            W_diff = np.linalg.norm(self.W - W_old) / np.linalg.norm(self.W)
            H1_diff = np.linalg.norm(self.H1 - H1_old) / np.linalg.norm(self.H1)
            H2_diff = np.linalg.norm(self.H2 - H2_old) / np.linalg.norm(self.H2)
            
            if max(W_diff, H1_diff, H2_diff) < 1e-10:
                break
        return self.W, self.H1, self.H2
        
    @staticmethod
    def normalize_smooth(x, eps = 1e-10):
        x += eps
        return x / x.sum()
    
    @staticmethod
    def js_divergence(p, q, epsilon=1e-10):
        def safe_log(x, epsilon=1e-10):
            return np.log(np.maximum(x, epsilon))
        def normalize_smooth(x, epsilon=1e-10):
            x = np.asarray(x, dtype=np.float64)
            if x.sum() == 0:
                x = np.ones_like(x)
            x = x + epsilon  
            return x / x.sum()
        q = np.asarray(q, dtype=np.float64)
        p = normalize_smooth(p, epsilon)
        q = normalize_smooth(q, epsilon)
        m = (p + q) / 2.0
        js_div = 0.5 * np.sum(p * (safe_log(p) - safe_log(m))) + \
                0.5 * np.sum(q * (safe_log(q) - safe_log(m)))
        if np.isnan(js_div):
            return 0.0 
            
        return max(0.0, js_div)


    def _get_common_activation(self):
        k = self.H1.shape[0]
        
        for i in range(k):
            act1 = self.H1[i,:]
            act2 = self.H2[i,:]
            
            js_div = CommonExtraction.js_divergence(act1, act2)
            self.patten_analysis.append(
                {
                    'pattern_idx': i,
                    'js_div': js_div,
                    'G1_activation': act1,
                    'G2_activation': act2
                }
            )
            
            
    def _identify_common_pattern(self):
        
        for pattern in self.patten_analysis:
            if pattern['js_div'] < 0.02:
                pattern['pattern_type'] = 'Common'
            else:
                pattern['pattern_type'] = 'Distinct'
                
    def _get_common_matrix(self) -> np.array:
        # 把H1和H2的对应的pattern_type的Common的激活模式平均得到H_common对应的部分，其余部分是0
        # 然后 *W 得到GCommon
        H_common = np.zeros_like(self.H1)
        for pattern in self.patten_analysis:
            if pattern['pattern_type'] == 'Common':
                H_common[pattern['pattern_idx'], :] = (pattern['G1_activation'] + pattern['G2_activation']) / 2
        G_common = np.dot(self.W, H_common)
        return G_common
        
def trans_triangular_to_full_matrix(triangular_values):
    matrix_size = 30
    full_matrix = np.zeros((matrix_size, matrix_size))
    tril_indices = np.tril_indices(matrix_size)
    full_matrix[tril_indices] = triangular_values
    full_matrix = full_matrix + full_matrix.T - np.diag(np.diag(full_matrix))
    return full_matrix

def visualize_lower_triangle(lower_triangle_values):
    full_matrix = trans_triangular_to_full_matrix(lower_triangle_values)
    plt.figure(figsize=(12, 10))
    sns.heatmap(full_matrix, 
                cmap='coolwarm',  
                center=0,         
                square=True,      
                annot=False)      
    
    plt.title('Edge Weight Visualization')
    plt.tight_layout()
    plt.show()


def normalize_matrix(m: torch.Tensor, symmetry: bool=True) -> torch.Tensor:
    """
    对邻接矩阵进行标准化处理
    Args:
        m (torch.Tensor): 输入邻接矩阵
        symmetry (bool): 是否进行对称化处理
    Returns:
        torch.Tensor: 标准化后的邻接矩阵
    """
    m = F.relu(m)
    if symmetry:
        m = m + torch.transpose(m, 0, 1)
    d = torch.sum(m, dim=1)
    d = torch.diag_embed(1 / torch.sqrt(d + 1e-10))
    # 计算标准化后的矩阵：D^(-1/2) A D^(-1/2)
    l = torch.matmul(torch.matmul(d, m), d)
    return l

def l1_regLoss(model, only=None, exclude=None):
    totalLoss = 0
    if only is None and exclude is None:
        for name, param in model.namded_parameters():
            totalLoss += torch.sum(torch.abs(param))
    elif only is not None:
        for name, param in model.namded_parameters():
            if name in only:
                totalLoss += torch.sum(torch.abs(param))
    elif exclude is not None:
        for name, param in model.namded_parameters():
            if name not in exclude:
                totalLoss += torch.sum(torch.abs(param))
    return totalLoss

def l2_regLoss(predict, label):
    if type(predict) == np.ndarry:
        numSamples = predict.shape[0]
    elif type(predict) == list:
        numSamples = len(predict)
        predict = np.array(predict)
        label = np.array(label)
    
    return np.sum(predict == label) / numSamples if numSamples > 0 else 0

def get_edge_weight() -> \
    Tuple[str, List[List[int]], List[List[float]]]:
    """
    返回edge idx 和 edge weight
    edge 是二维数组，分别表示每个电极和彼此之间的连接
    edge idx 是一个二维数组 表示电极的索引
    例如 edge_idx[0] = [0,1,2,3...]
    edge_idx[1] = [0,1,2,3...]
    edge_weight 是一个二维数组 表示电极之间的连接权重
    两个电极如果是同一个 则连接权重为1
    否则为两个电极之间的距离的平方
    delta是一个常数，用于控制连接的稀疏程度
    如果两个电极之间的距离的平方小于delta，则连接权重为0
    否则为exp(-距离的平方/2) 这表示连接距离越远 权重越小 
    为什么使用指数函数可能是因为更平滑的变形
    """
    total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 Fc1 Fc2 Fc5 Fc6 Cz C3 C4 T7 T8 Cp1 Cp2 Cp5 Cp6 Pz P3 P4 P7 P8 Po3 Po4 Oz O1 O2'''.split()
    raw_pos_value = np.load('/home/pekew/code/EEGNet/models/pos.npy') * 100
    # raw_pos_value 有32个位置，最后两个是参考电极 不用考虑
    edge_pos_value = {
        name: raw_pos_value[idx]
        for idx, name in enumerate(total_part)
    }
    edge_weight = np.zeros([len(total_part), len(total_part)])
    delta = 2
    edge_index = [[], []]

    for (i, electrode1), (j, electrode2) in product(enumerate(total_part), enumerate(total_part)):
        edge_index[0].append(i)
        edge_index[1].append(j)

        if i==j:
            edge_weight[i][j] = 1
        else:
            pos1 = edge_pos_value[electrode1]
            pos2 = edge_pos_value[electrode2]
            edge_weight[i][j] = np.sum((pos1 - pos2) ** 2)

            if delta / edge_weight[i][j] > 1:
                edge_weight[i][j] = math.exp(-edge_weight[i][j] / 2)
            else:
                pos1 = edge_pos_value[electrode1]
                pos2 = edge_pos_value[electrode2]
                dist = np.sum((pos1 - pos2) ** 2)
                edge_weight[i][j] = math.exp(-dist / (2 * delta))  # 使用高斯核
    return total_part, edge_index, edge_weight


def average_by_regions(matrix, electrode_names, regions_mapping):
    """将电极级别的连接矩阵转换为脑区级别的平均连接矩阵"""
    region_names = list(regions_mapping.keys())
    n_regions = len(region_names)
    region_matrix = np.zeros((n_regions, n_regions))
    
    electrode_to_region_idx = {}
    for i, name in enumerate(electrode_names):
        for region_idx, (region, electrodes) in enumerate(regions_mapping.items()):
            if name in electrodes:
                electrode_to_region_idx[i] = region_idx
                break
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            region_i = electrode_to_region_idx[i]
            region_j = electrode_to_region_idx[j]
            region_matrix[region_i, region_j] += matrix[i, j]
    
    for i in range(n_regions):
        for j in range(n_regions):
            n_connections = len(regions_mapping[region_names[i]]) * len(regions_mapping[region_names[j]])
            region_matrix[i, j] /= n_connections
            
    return region_matrix, region_names

def create_region_graph(region_matrix, region_names):
    """创建脑区级别的图结构"""
    G = nx.from_numpy_array(region_matrix)
    
    colors = {
        'Frontal': '#4299e1', 
        'Frontocentral': '#f56565', 
        'Central': '#9f7aea',    
        'Temporal': '#48bb78',   
        'Centroparietal': '#ed64a6',
        'Parietal': '#ecc94b', 
        'Parietooccipital': '#667eea', 
        'Occipital': '#ed8936'
    }
    
    for i, region in enumerate(region_names):
        G.nodes[i]['region'] = region
        G.nodes[i]['color'] = colors[region]
        G.nodes[i]['name'] = region
    
    return G

def plot_region_connectogram(G, threshold=0.0001):
    """绘制脑区级别的连接图"""
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    n_nodes = len(G.nodes())
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    radius = 1
    pos = {i: (radius * np.cos(angle), radius * np.sin(angle)) 
            for i, angle in enumerate(angles)}
    
    def create_segment(start_angle, end_angle, radius, color):
        theta = np.linspace(start_angle, end_angle, 50)
        inner_radius = radius * 0.9
        outer_radius = radius * 1.15
        
        vertices = [(inner_radius * np.cos(t), inner_radius * np.sin(t)) for t in theta]
        vertices.extend([(outer_radius * np.cos(t), outer_radius * np.sin(t)) 
                        for t in theta[::-1]])
        return patches.Polygon(vertices, facecolor=color, alpha=0.3, edgecolor='none')
    
    segment_width = 2*np.pi / n_nodes
    for i, node in enumerate(G.nodes()):
        start_angle = angles[i] - segment_width/2
        end_angle = angles[i] + segment_width/2
        color = G.nodes[node]['color']
        segment = create_segment(start_angle, end_angle, radius, color)
        ax.add_patch(segment)
    
    for (u, v, w) in G.edges(data=True):
        if abs(w['weight']) > threshold:
            start = np.array(pos[u])
            end = np.array(pos[v])
            diff = end - start
            dist = np.linalg.norm(diff)
            mid_point = (start + end) / 2
            perp = np.array([-diff[1], diff[0]])
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-6:
                perp = perp / perp_norm
                curvature = 0.2 + 0.3 * (dist/2)
                control_point = mid_point + curvature * perp
                verts = [tuple(start), tuple(control_point), tuple(end)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                color = G.nodes[u]['color']
                alpha = min(abs(w['weight'])*90, 0.8)
                line_width = abs(w['weight'])*120
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                                        alpha=alpha, linewidth=line_width)
                ax.add_patch(patch)
    
    for i in G.nodes():
        ax.plot(pos[i][0], pos[i][1], 'o', color=G.nodes[i]['color'],
                markersize=8, zorder=3)
        angle = angles[i]
        label_radius = radius * 1.2
        label_pos = (label_radius * np.cos(angle), label_radius * np.sin(angle))
        ha = 'left' if -np.pi/2 <= angle <= np.pi/2 else 'right'
        va = 'center'
        rotation = np.degrees(angle)
        if ha == 'right':
            rotation += 180
        if rotation > 90 and rotation <= 270:
            rotation -= 180
            
        plt.text(label_pos[0], label_pos[1], G.nodes[i]['name'],
                ha=ha, va=va, rotation=rotation,
                rotation_mode='anchor', fontsize=10)

    plt.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.title('Brain Region Connectogram', pad=20)
    
    return plt

def visualize_brain_regions(matrix,threshold):
    
    region_matrix, region_names = average_by_regions(matrix, total_part, regions_mapping)
    G = create_region_graph(region_matrix, region_names)
    return plot_region_connectogram(G, threshold)