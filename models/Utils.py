import torch
import numpy as np
import math
from typing import List, Tuple
from itertools import product
from torch.nn import functional as F
from sklearn.decomposition import NMF
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.path import Path
from Utils.Config import Args
from scipy.optimize import linear_sum_assignment
import matplotlib.patches as patches

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
        # 8个功能脑区的划分索引
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

class Preprocessor:
    """预处理类"""
    def __init__(self, input_data: DeidentificationInput):
        self.module_indices = input_data.module_indices

    def normalize_matrix(self, W: np.ndarray) -> np.ndarray:
        """归一化到[0,1]范围"""
        return (W - W.min()) / (W.max() - W.min())
    
    def ensure_symmetry(self, W: np.ndarray) -> np.ndarray:
        """确保矩阵对称性"""
        return (W + W.T) / 2
    
    def initialize_module_structure(self) -> np.ndarray:
        """初始化模块结构掩码"""
        n = 30
        mask = np.zeros((n, n))
        for module in self.module_indices:
            mask[np.ix_(module, module)] = 1
        return mask

    def preprocess(self, W: np.ndarray) -> np.ndarray:
        """执行完整的预处理流程"""
        W = self.normalize_matrix(W)
        W = self.ensure_symmetry(W)
        return W

class ModularNMF:
    """模块化NMF分解类"""
    def __init__(self, n_components: int = 10):
        self.n_components = n_components
    
    def _initialize_with_modules(self, module_mask: np.ndarray) -> np.ndarray:
        """考虑模块结构的初始化"""
        n = module_mask.shape[0]
        H_init = np.random.rand(n, self.n_components) * 0.2
        H_init = H_init * module_mask.mean(axis=1).reshape(-1, 1)
        return H_init

    def decompose(self, W: np.ndarray, module_mask: np.ndarray) -> np.ndarray:
        """执行NMF分解"""
        H_init = self._initialize_with_modules(module_mask)
        
        model = NMF(
            n_components=self.n_components,
            init='random',  # 使用随机初始化
            solver='mu',    # 使用乘性更新规则
            max_iter=800,
            random_state=Args.rand_seed
        )
        H = model.fit_transform(W, H=H_init, W=None)
        return H

class BasisAnalyzer:
    """基底分析类"""
    def __init__(self, module_indices):
        self.module_indices = module_indices

    def _compute_module_contributions(self, basis_vector: np.ndarray) -> np.ndarray:
        """计算基底在各模块的贡献"""
        contributions = []
        for module in self.module_indices:
            contribution = np.mean(basis_vector[module])
            contributions.append(contribution)
        return np.array(contributions)

    def _identify_pattern(self, contributions: np.ndarray) -> str:
        """识别基底的连接模式"""
        max_contrib_idx = np.argmax(contributions)
        if contributions[max_contrib_idx] > 0.5:
            return f"Module-{max_contrib_idx}-dominant"
        return "Distributed"

    def analyze_patterns(self, H: np.ndarray) -> list:
        """分析基底的连接模式"""
        patterns = []
        for i in range(H.shape[1]):
            contributions = self._compute_module_contributions(H[:, i])
            pattern_type = self._identify_pattern(contributions)
            patterns.append(pattern_type)
        return patterns

    def _compute_similarity(self, H1: np.ndarray, H2: np.ndarray) -> np.ndarray:
        """计算两组基底间的相似度"""
        n_bases = H1.shape[1]
        similarity_matrix = np.zeros((n_bases, n_bases))
        
        for i in range(n_bases):
            for j in range(n_bases):
                similarity_matrix[i, j] = 1 - cosine(H1[:, i], H2[:, j])
        
        return similarity_matrix

    def align_bases(self, H1: np.ndarray, H2: np.ndarray) -> tuple:
        """对齐两组基底"""
        similarity_matrix = self._compute_similarity(H1, H2)
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        return row_ind, col_ind

class Reconstructor:
    """重构类"""
    def _fuse_bases(self, H1: np.ndarray, H2: np.ndarray, matches: tuple) -> np.ndarray:
        """融合对齐的基底"""
        row_ind, col_ind = matches
        H_common = np.zeros_like(H1)
        
        for i, j in zip(row_ind, col_ind):
            H_common[:, i] = (H1[:, i] + H2[:, j]) / 2
            
        return H_common

    def optimize_structure(self, W: np.ndarray, module_mask: np.ndarray) -> np.ndarray:
        """优化模块结构"""
        # 增强模块内连接
        W = W * (1 + 0.1 * module_mask)
        # 抑制模块间连接
        W = W * (1 - 0.62 * (1 - module_mask))
        # 添加稀疏化处理
        threshold = 0.315  # 设置阈值
        W[W < threshold] = 0  # 将小于阈值的连接置零
        # 确保值在[0,1]范围内
        W = np.clip(W, 0, 1)
        
        return W

    def fuse_and_reconstruct(self, H1: np.ndarray, H2: np.ndarray, matches: tuple) -> np.ndarray:
        """融合基底并重构"""
        H_common = self._fuse_bases(H1, H2, matches)
        W_reconstructed = np.dot(H_common, H_common.T)
        return W_reconstructed

class Validator:
    """验证类"""
    def _compute_similarity_metrics(self, W1: np.ndarray, W2: np.ndarray, W_common: np.ndarray) -> dict:
        """计算相似度指标"""
        sim1 = 1 - np.mean(np.abs(W1 - W_common))
        sim2 = 1 - np.mean(np.abs(W2 - W_common))
        return {
            'similarity_to_W1': sim1,
            'similarity_to_W2': sim2
        }

    def _compute_modularity_metrics(self, W: np.ndarray, module_mask: np.ndarray) -> float:
        """计算模块化指标"""
        module_density = np.sum(W * module_mask) / np.sum(module_mask)
        non_module_density = np.sum(W * (1 - module_mask)) / np.sum(1 - module_mask)
        return module_density / (non_module_density + 1e-10)

    def _compute_sparsity_metrics(self, W: np.ndarray) -> float:
        """计算稀疏度指标"""
        return 1 - np.count_nonzero(W > 0.1) / W.size

    def evaluate_results(self, W1: np.ndarray, W2: np.ndarray, W_common: np.ndarray, 
                        module_mask: np.ndarray) -> dict:
        """评估去个性化效果"""
        metrics = {
            **self._compute_similarity_metrics(W1, W2, W_common),
            'modularity': self._compute_modularity_metrics(W_common, module_mask),
            'sparsity': self._compute_sparsity_metrics(W_common)
        }
        return metrics

class ComprehensiveDeidentification:
    """完整的去个性化处理类"""
    def __init__(self, n_components: int = 10):
        self.input_data = DeidentificationInput()
        self.preprocessor = Preprocessor(self.input_data)
        self.nmf = ModularNMF(n_components)
        self.analyzer = BasisAnalyzer(self.input_data.module_indices)
        self.reconstructor = Reconstructor()
        self.validator = Validator()
    
    def perform_deidentification(self, W1: np.ndarray, W2: np.ndarray) -> tuple:
        W1 = self.preprocessor.preprocess(W1)
        W2 = self.preprocessor.preprocess(W2)
        module_mask = self.preprocessor.initialize_module_structure()
        H1 = self.nmf.decompose(W1, module_mask)
        H2 = self.nmf.decompose(W2, module_mask)
        patterns = self.analyzer.analyze_patterns(H1)
        matches = self.analyzer.align_bases(H1, H2)
        W_common = self.reconstructor.fuse_and_reconstruct(H1, H2, matches)
        W_common = self.reconstructor.optimize_structure(W_common, module_mask)
        metrics = self.validator.evaluate_results(W1, W2, W_common, module_mask)
        return W_common, metrics, patterns

def matrix_to_region_matrix(matrix, electrode_names, regions_mapping):
    """将电极级别的连接矩阵转换为脑区级别的连接矩阵
    Args:
        matrix: 电极级别的连接矩阵 (30x30)
        electrode_names: 电极名称列表
        regions_mapping: 脑区映射字典
    Returns:
        region_matrix: 脑区级别的连接矩阵
        region_names: 脑区名称列表
    """
    region_names = list(regions_mapping.keys())
    n_regions = len(region_names)
    region_matrix = np.zeros((n_regions, n_regions))
    
    # 创建电极到脑区的映射字典
    electrode_to_region = {}
    for region, electrodes in regions_mapping.items():
        for electrode in electrodes:
            electrode_to_region[electrode] = region
    
    # todo bug
    # 计算每个脑区对之间的平均连接强度
    for i, region1 in enumerate(region_names):
        for j, region2 in enumerate(region_names):
            # 获取属于这两个脑区的电极索引
            electrodes1 = [k for k, name in enumerate(electrode_names) 
                         if name in regions_mapping[region1]]
            electrodes2 = [k for k, name in enumerate(electrode_names) 
                         if name in regions_mapping[region2]]
            
            # 计算这两个脑区之间所有电极对的平均连接强度
            connections = []
            for e1 in electrodes1:
                for e2 in electrodes2:
                    if i != j or e1 < e2:  # 避免重复计算
                        connections.append(matrix[e1, e2])
            
            region_matrix[i, j] = np.mean(connections)
            region_matrix[j, i] = region_matrix[i, j]  # 确保矩阵对称
    
    return region_matrix, region_names

def plot_region_connectogram(matrix, region_names, threshold=0.0001):
    """
    Creates a circular connectogram visualization of brain regions.
    
    Parameters:
    matrix : numpy.ndarray
        Matrix of connection strengths between brain regions
    region_names : list
        List of region names
    threshold : float
        Minimum absolute weight value to show connections
    """
    # 创建NetworkX图
    G = nx.Graph()
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
    
    # 添加节点和边
    for i, region in enumerate(region_names):
        G.add_node(i, region=region, color=colors[region], name=region)
        for j in range(i+1, len(region_names)):
            if abs(matrix[i, j]) > threshold:
                G.add_edge(i, j, weight=matrix[i, j])
    
    # 绘图部分与原来类似，但简化了标签
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    n_nodes = len(G.nodes())
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    radius = 1
    pos = {i: (radius * np.cos(angle), radius * np.sin(angle)) 
           for i, angle in enumerate(angles)}
    
    # 绘制区域段
    segment_width = 2*np.pi / n_nodes
    for i, node in enumerate(G.nodes()):
        start_angle = angles[i] - segment_width/2
        end_angle = angles[i] + segment_width/2
        theta = np.linspace(start_angle, end_angle, 50)
        inner_radius = radius * 0.9
        outer_radius = radius * 1.15
        
        vertices = [(inner_radius * np.cos(t), inner_radius * np.sin(t)) for t in theta]
        vertices.extend([(outer_radius * np.cos(t), outer_radius * np.sin(t)) 
                        for t in theta[::-1]])
        ax.add_patch(patches.Polygon(vertices, facecolor=G.nodes[node]['color'], 
                                   alpha=0.3, edgecolor='none'))
    
    # 绘制连接
    for (u, v, w) in G.edges(data=True):
        start = np.array(pos[u])
        end = np.array(pos[v])
        diff = end - start
        dist = np.linalg.norm(diff)
        mid_point = (start + end) / 2
        perp = np.array([-diff[1], diff[0]])
        perp = perp / np.linalg.norm(perp)
        curvature = 0.2 + 0.3 * (dist/2)
        control_point = mid_point + curvature * perp
        
        path = Path([tuple(start), tuple(control_point), tuple(end)],
                   [Path.MOVETO, Path.CURVE3, Path.CURVE3])
        alpha = min(abs(w['weight'])*300, 0.8)
        line_width = abs(w['weight'])*500
        patch = patches.PathPatch(path, facecolor='none', 
                                edgecolor=G.nodes[u]['color'],
                                alpha=alpha, linewidth=line_width)
        ax.add_patch(patch)
    
    # 绘制节点和标签
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
    plt.title('Brain Region Connectivity', pad=20)
    
    return plt
def plot_brain_connectivity_nilearn(matrix, region_names, threshold=0.0001):
    
    
    """使用Nilearn绘制脑区连接图
    
    Args:
        matrix: 脑区连接矩阵
        region_names: 脑区名称列表
        threshold: 连接强度阈值
    
    Returns:
        Nilearn绘图对象
    """
    from nilearn import plotting
    import pandas as pd
    
    
    coords = {
        'Frontal': (30, 45, 40),           # 前额叶，偏右侧
        'Frontocentral': (25, 30, 55),     # 额中央区，稍偏右
        'Central': (20, 0, 65),            # 中央区，轻微偏右
        'Temporal': (50, -10, 0),          # 颞叶，明显偏右侧
        'Centroparietal': (20, -30, 65),   # 中央顶区，轻微偏右
        'Parietal': (25, -60, 55),         # 顶叶，稍偏右
        'Parietooccipital': (30, -75, 40), # 顶枕区，偏右侧
        'Occipital': (35, -95, 20)         # 枕叶，偏右侧
    }
    
    # 创建节点坐标数组
    node_coords = np.array([coords[region] for region in region_names])
    
    # 创建连接对列表
    connections = []
    for i in range(len(region_names)):
        for j in range(i + 1, len(region_names)):
            if abs(matrix[i, j]) > threshold:
                connections.append((
                    i, j, 
                    matrix[i, j]
                ))
    
    # 转换为pandas DataFrame
    connections_df = pd.DataFrame(connections, columns=['source', 'target', 'weight'])
    
    # 创建颜色映射
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
    node_colors = [colors[region] for region in region_names]
    
    # 创建节点大小列表（可以基于连接强度调整）
    node_sizes = [np.sum(np.abs(matrix[i])) * 100 for i in range(len(region_names))]
    
    # 创建绘图
    display = plotting.plot_connectome(
        adjacency_matrix=matrix,
        node_coords=node_coords,
        node_color=node_colors,
        node_size=node_sizes,
        edge_threshold=threshold,
        edge_cmap='RdBu_r',
        edge_vmin=-np.max(np.abs(matrix)),
        edge_vmax=np.max(np.abs(matrix)),
        title='Brain Connectivity Visualization',
        colorbar=True
    )
    
    # 添加球体表示节点
    for i, (coord, size, color) in enumerate(zip(node_coords, node_sizes, node_colors)):
        display.add_markers(
            [coord],
            marker_color=color,
            marker_size=size
        )
    
    
    return display
def plot_brain_connectivity_multiple_views(matrix, region_names, threshold=0.0001):
    """绘制多视角的大脑连接图
    
    Args:
        matrix: 脑区连接矩阵
        region_names: 脑区名称列表
        threshold: 连接强度阈值
    """
    from nilearn import plotting
    import matplotlib.pyplot as plt
    
    
    coords = {
        'Frontal': (30, 45, 40),           # 前额叶，偏右侧
        'Frontocentral': (25, 30, 55),     # 额中央区，稍偏右
        'Central': (20, 0, 65),            # 中央区，轻微偏右
        'Temporal': (50, -10, 0),          # 颞叶，明显偏右侧
        'Centroparietal': (20, -30, 65),   # 中央顶区，轻微偏右
        'Parietal': (25, -60, 55),         # 顶叶，稍偏右
        'Parietooccipital': (30, -75, 40), # 顶枕区，偏右侧
        'Occipital': (35, -95, 20)         # 枕叶，偏右侧
    }
    
    # 创建节点坐标数组
    node_coords = np.array([coords[region] for region in region_names])
    
    # 创建颜色映射
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
    node_colors = [colors[region] for region in region_names]
    
    # 固定节点大小
    node_size = 100
    
    # 计算连接强度的范围用于线宽映射
    max_weight = np.max(np.abs(matrix))
    min_weight = np.min(np.abs(matrix))
    
    edge_threshold = "55.3%"
    
    # 创建线宽映射函数
    def get_line_width(weight):
        # 将连接强度映射到1-8的范围内
        scale = (abs(weight) - min_weight) / (max_weight - min_weight)
        return 1 + 10 * scale

    # 修改矩阵，将小于阈值的连接设为0
    matrix_thresholded = matrix.copy()
    # matrix_thresholded[np.abs(matrix_thresholded) < threshold] = 0
    
    # 创建图形
    fig = plt.figure(figsize=(20, 6))
    
    # 左视图
    ax1 = plt.subplot(131)
    plotting.plot_connectome(
        adjacency_matrix=matrix_thresholded,
        node_coords=node_coords,
        node_color=node_colors,
        node_size=node_size,
        edge_threshold=edge_threshold,
        edge_cmap='RdBu_r',
        edge_vmin=-max_weight*0.8,
        edge_vmax=max_weight*0.8,
        display_mode='x',
        title='Left View',
        axes=ax1,
    )
    
    # 顶视图
    ax2 = plt.subplot(132)
    plotting.plot_connectome(
        adjacency_matrix=matrix_thresholded,
        node_coords=node_coords,
        node_color=node_colors,
        node_size=node_size,
        edge_threshold=edge_threshold,
        edge_cmap='RdBu_r',
        edge_vmin=-max_weight,
        edge_vmax=max_weight,
        display_mode='z',
        title='Top View',
        axes=ax2,
    )
    
    # 前视图
    ax3 = plt.subplot(133)
    plotting.plot_connectome(
        adjacency_matrix=matrix_thresholded,
        node_coords=node_coords,
        node_color=node_colors,
        node_size=node_size,
        edge_threshold=edge_threshold,
        edge_cmap='RdBu_r',
        edge_vmin=-max_weight,
        edge_vmax=max_weight,
        display_mode='y',
        title='Front View',
        axes=ax3,
        colorbar=False,
    )
    
    # 添加图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                label=region,
                                markerfacecolor=color, markersize=8)
                      for region, color in zip(region_names, node_colors)]
    
    # 添加连接强度图例
    for weight in np.linspace(min_weight, max_weight, 3):
        legend_elements.append(plt.Line2D([0], [0], color='gray', 
                                        label=f'Strength: {weight:.3f}',
                                        linewidth=get_line_width(weight)))
    
    fig.legend(handles=legend_elements, 
              loc='center right', 
              bbox_to_anchor=(0.98, 0.5),
              title='Brain Regions and Connection Strengths')
    
    plt.tight_layout()
    return fig

def plot_connectogram(G, threshold=0.0001):
    """
    Creates a circular connectogram visualization of brain regions and their connections.
    
    Parameters:
    G : networkx.Graph
        Graph with nodes representing brain regions and edges representing connections
    threshold : float
        Minimum absolute weight value to show connections
    """
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
            if perp_norm > 1e-6:  # Check for non-zero norm
                perp = perp / perp_norm
                curvature = 0.2 + 0.3 * (dist/2)
                control_point = mid_point + curvature * perp
                verts = [tuple(start), tuple(control_point), tuple(end)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                color = G.nodes[u]['color']
                alpha = min(abs(w['weight'])*300, 0.8)
                line_width = abs(w['weight'])*500
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                                        alpha=alpha, linewidth=line_width)
                ax.add_patch(patch)
    
    for i in G.nodes():
        ax.plot(pos[i][0], pos[i][1], 'o', color=G.nodes[i]['color'],
                markersize=6, zorder=3)
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
                rotation_mode='anchor', fontsize=8)

    plt.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.title('Brain Region Connectogram', pad=20)
    
    return plt

def matrix_to_connectogram_data(matrix, electrode_names, regions_mapping):
    """将邻接矩阵转换为环形图数据结构
    Args:
        matrix: 邻接矩阵
        electrode_names: 电极名称列表
        regions_mapping: 脑区映射字典
    """
    # 创建图
    if matrix.shape != (30, 30):
        matrix = trans_triangular_to_full_matrix(matrix)
    G = nx.from_numpy_array(matrix)
    
    # 添加节点属性
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
    
    for i, name in enumerate(electrode_names):
        for region, electrodes in regions_mapping.items():
            if name in electrodes:
                G.nodes[i]['region'] = region
                G.nodes[i]['color'] = colors[region]
                G.nodes[i]['name'] = name
                break
    return G


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