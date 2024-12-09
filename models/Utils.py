import torch
import numpy as np
import math
from itertools import product
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.path import Path
import matplotlib.patches as patches
from sklearn.preprocessing import normalize
from dataclasses import dataclass
from numpy.typing import NDArray
from typing import LiteralString, Optional, Tuple, List

total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 Fc1 Fc2 Fc5 Fc6 Cz C3 C4 T7 T8 Cp1 Cp2 Cp5 Cp6 Pz P3 P4 P7 P8 Po3 Po4 Oz O1 O2'''.split()
regions_mapping = {
    'Frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'Fz'],
    'Frontocentral': ['Fc1', 'Fc2', 'Fc5', 'Fc6'],
    'Central': ['C3', 'C4', 'Cz'],
    'Temporal': ['T7', 'T8'], 
    'Centroparietal': ['Cp1', 'Cp2', 'Cp5', 'Cp6'],
    'Parietal': ['P3', 'P4', 'P7', 'P8', 'Pz'],
    'Parietooccipital': ['Po3', 'Po4'], 
    'Occipital': ['O1', 'O2', 'Oz']
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


def normalize_matrix_01(W: np.ndarray) -> np.ndarray:
    return (W - W.min()) / (W.max() - W.min())

def ensure_symmetry(W: np.ndarray) -> np.ndarray:
    return (W + W.T) / 2

def initialize_module_structure(module_indices) -> np.ndarray:
    n = 30
    mask = np.zeros((n, n))
    for module in module_indices:
        mask[np.ix_(module, module)] = 1
    return mask

@dataclass
class Pattern:
    """Data class for pattern analysis results"""
    pattern_idx: int
    js_div: float
    G1_activation: NDArray
    G2_activation: NDArray
    pattern_type: Optional[str] = None

class CommonExtractionError(Exception):
    """Base exception class for CommonExtraction errors"""
    pass

class ConvergenceError(CommonExtractionError):
    """Raised when the algorithm fails to converge"""
    pass

class CommonExtraction:
    """
    Common feature extraction using Non-negative Matrix Factorization
    Solves the optimization problem:
    min ||G1 - W * H1||^2 + ||G2 - W * H2||^2
    s.t. W >= 0, H1 >= 0, H2 >= 0
    """
    
    def __init__(self, G1: NDArray, G2: NDArray, max_iter: int = 500, eps: float = 1e-10) -> None:
        if G1.shape != G2.shape:
            raise ValueError("Input matrices G1 and G2 must have the same shape")
            
        self.eps = eps
        self.G1 = G1
        self.G2 = G2
        self.max_iter = max_iter
        
        # Initialize matrices
        shape = G1.shape
        self.W = self._initialize_matrix(shape)
        self.H1 = self._initialize_matrix(shape)
        self.H2 = self._initialize_matrix(shape)
        
        self.pattern_analysis: List[Pattern] = []
        
    @staticmethod
    def _initialize_matrix(shape: Tuple[int, ...]) -> NDArray:
        matrix = np.random.rand(shape[0], shape[1])
        return normalize(matrix, axis=0)
    
    @staticmethod
    def normalize_smooth(x: NDArray, eps: float = 1e-10) -> NDArray:
        x = np.asarray(x, dtype=np.float64)
        x = np.maximum(x, eps)
        return x / np.sum(x)
    
    def _update_matrices(self) -> Tuple[float, float, float]:
        W_old = self.W.copy()
        H1_old = self.H1.copy()
        H2_old = self.H2.copy()
        
        numerator = self.G1 @ self.H1.T + self.G2 @ self.H2.T
        denominator = self.W @ (self.H1 @ self.H1.T + self.H2 @ self.H2.T)
        self.W *= numerator / (denominator + self.eps)
        numerator = self.W.T @ self.G1
        denominator = (self.W.T @ self.W) @ self.H1
        self.H1 *= numerator / (denominator + self.eps)
        numerator = self.W.T @ self.G2
        denominator = (self.W.T @ self.W) @ self.H2
        self.H2 *= numerator / (denominator + self.eps)
        
        return (
            float(np.linalg.norm(self.W - W_old) / np.linalg.norm(self.W)),
            float(np.linalg.norm(self.H1 - H1_old) / np.linalg.norm(self.H1)),
            float(np.linalg.norm(self.H2 - H2_old) / np.linalg.norm(self.H2))
        )
    
    def _get_common_base(self) -> Tuple[NDArray, NDArray, NDArray]:
        for iteration in range(self.max_iter):
            try:
                W_diff, H1_diff, H2_diff = self._update_matrices()
                
                if max(W_diff, H1_diff, H2_diff) < self.eps:
                    return self.W, self.H1, self.H2
                    
            except np.linalg.LinAlgError as e:
                raise ConvergenceError(f"Linear algebra error at iteration {iteration}") from e
                
        raise ConvergenceError(f"Failed to converge after {self.max_iter} iterations")
    
    def analyze_patterns(self, js_threshold: float = 0.02) -> None:
        self._get_common_activation()
        self._identify_common_pattern(js_threshold)
    
    @ staticmethod
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
    
    def _get_common_activation(self) -> None:
        for i in range(self.H1.shape[0]):
            self.pattern_analysis.append(Pattern(
                pattern_idx=i,
                js_div=self.js_divergence(self.H1[i, :], self.H2[i, :]),
                G1_activation=self.H1[i, :],
                G2_activation=self.H2[i, :]
            ))
    
    def _identify_common_pattern(self, threshold: float ) -> None:
        for pattern in self.pattern_analysis:
            pattern.pattern_type = 'Common' if pattern.js_div < threshold else 'Distinct'
    
    def get_common_matrix(self) -> NDArray:
        H_common = np.zeros_like(self.H1)
        
        for pattern in self.pattern_analysis:
            if pattern.pattern_type == 'Common':
                H_common[pattern.pattern_idx, :] = (
                    pattern.G1_activation + pattern.G2_activation
                ) / 2
                
        return self.W @ H_common


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


def normalize_matrix(m: torch.Tensor, symmetry: bool = True) -> torch.Tensor:
    """
    normalization the adjacency matrix
    Args:
        m (torch.Tensor): input adjacency matrix
        symmetry (bool): whether the matrix is symmetric
    Returns:
        torch.Tensor: normalized adjacency matrix
    """
    m = F.relu(m)
    if symmetry:
        m = m + torch.transpose(m, 0, 1)
    d = torch.sum(m, dim=1)
    d = torch.diag_embed(1 / torch.sqrt(d + 1e-10))
    # D^(-1/2) A D^(-1/2)
    return torch.matmul(torch.matmul(d, m), d)


def l1_reg_loss(model: torch.nn.Module, only: Optional[List[str]] = None, exclude: Optional[List[str]] = None) -> torch.Tensor:

    if only is not None and exclude is not None:
        raise ValueError("'only' and 'exclude' parameters cannot be specified simultaneously")
        
    total_loss = torch.tensor(0)
    for name, param in model.named_parameters():
        if only is not None:
            if name in only:
                total_loss += torch.sum(torch.abs(param))
        elif exclude is not None:
            if name not in exclude:
                total_loss += torch.sum(torch.abs(param))
        else:
            total_loss += torch.sum(torch.abs(param))
    
    return total_loss


def l2_reg_loss(predict, label):
    if type(predict) is np.ndarray:
        numSamples = predict.shape[0]
    else: # list
        numSamples = len(predict)
        predict = np.array(predict)
        label = np.array(label)

    return np.sum(predict == label) / numSamples if numSamples > 0 else 0

def get_edge_weight(
    pos_file: str = '/home/pekew/code/EEGNet/models/pos.npy',
    delta: float = 2.0,
    scale: float = 100.0
) -> Tuple[List[LiteralString], List[List[int]], np.ndarray]:
    """
    Calculate edge indices and weights for EEG electrode connections.
    Args:
        pos_file (str): Path to the numpy file containing electrode positions
        delta (float): Distance threshold for controlling connection sparsity
        scale (float): Scaling factor for position values
    Returns:
        Tuple containing:
            - List[str]: List of electrode names
            - List[List[int]]: Edge indices as [source_indices, target_indices]
            - np.ndarray: Edge weights matrix
            
    Notes:
        Weight calculation:
        - If source == target: weight = 1
        - If distance²/delta > 1: weight = exp(-distance²/(2*delta))
        - Otherwise: weight = exp(-distance²/2)
    """
    total_part = '''Fp1 Fp2 Fz F3 F4 F7 F8 Fc1 Fc2 Fc5 Fc6 Cz C3 C4 T7 T8 Cp1
    Cp2 Cp5 Cp6 Pz P3 P4 P7 P8 Po3 Po4 Oz O1 O2'''.split()
    
    raw_pos_value = np.load(pos_file) * scale
    edge_pos_value = {
        name: raw_pos_value[idx]
        for idx, name in enumerate(total_part)
    }
    
    num_electrodes = len(total_part)
    edge_weight = np.zeros([num_electrodes, num_electrodes])
    edge_index = [[], []]

    for (i, electrode1), (j, electrode2) in product(enumerate(total_part), enumerate(total_part)):
        edge_index[0].append(i)
        edge_index[1].append(j)

        if i == j:
            edge_weight[i][j] = 1.0
            continue
            
        pos1 = edge_pos_value[electrode1]
        pos2 = edge_pos_value[electrode2]
        dist_squared = np.sum((pos1 - pos2) ** 2)
        
        if dist_squared > delta:
            edge_weight[i][j] = np.exp(-dist_squared / (2 * delta))
        else:
            edge_weight[i][j] = np.exp(-dist_squared / 2)
            
    return total_part, edge_index, edge_weight


def average_by_regions(matrix, electrode_names, regions_mapping):
    """
        transform electrode-level connection matrix to average region-level connection matrix
    """
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
            n_connections = len(
                regions_mapping[region_names[i]]) * len(regions_mapping[region_names[j]])
            region_matrix[i, j] /= n_connections
    return region_matrix, region_names


def create_region_graph(region_matrix, region_names):
    """create a graph from region-level connection matrix"""
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
    """draw a connectogram of brain regions"""
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

        vertices = [(inner_radius * np.cos(t), inner_radius * np.sin(t))
                    for t in theta]
        vertices.extend([(outer_radius * np.cos(t), outer_radius * np.sin(t))
                        for t in theta[::-1]])
        return patches.Polygon(vertices, facecolor=color, alpha=0.3, edgecolor='none')

    segment_width = 2*np.pi / n_nodes
    for i, node in enumerate(G.nodes()):
        start_angle = angles[i] - segment_width/2
        end_angle = angles[i] + segment_width/2
        color = G.nodes[node]['color']
        segment = create_segment(
            start_angle, end_angle, radius, color)
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
        label_pos = (label_radius * np.cos(angle),
                    label_radius * np.sin(angle))
        ha = 'left' if -np.pi/2 <= angle <= np.pi/2 else 'right'
        va = 'center'
        rotation = np.degrees(angle)
        if ha == 'right':
            rotation += 180
        if rotation > 90 and rotation <= 270:
            rotation -= 180

        plt.text(label_pos[0].item(), label_pos[1].item(), G.nodes[i]['name'],
                ha=ha, va=va, rotation=rotation,
                rotation_mode='anchor', fontsize=10)

    plt.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.title('Brain Region Connectogram', pad=20)

    return plt


def visualize_brain_regions(matrix, threshold):
    region_matrix, region_names = average_by_regions(
        matrix, total_part, regions_mapping)
    G = create_region_graph(region_matrix, region_names)
    return plot_region_connectogram(G, threshold)
