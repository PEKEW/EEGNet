import torch
import Utils
import Utils.Config
from models.Utils import get_data_loaders_gender
import models.trainer as Trainer
from models.DGCNN import DGCNN
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.patches as patches
from matplotlib.path import Path
from sklearn.decomposition import NMF
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine


# todo trans these to a new file
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
        H_init = np.random.rand(n, self.n_components) * 0.1
        # 使用模块掩码调整初始值
        H_init = H_init * module_mask.mean(axis=1).reshape(-1, 1)
        return H_init

    def decompose(self, W: np.ndarray, module_mask: np.ndarray) -> np.ndarray:
        """执行NMF分解"""
        H_init = self._initialize_with_modules(module_mask)
        
        model = NMF(
            n_components=self.n_components,
            init='random',  # 使用随机初始化
            solver='mu',    # 使用乘性更新规则
            max_iter=200,
            random_state=42
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
        W = W * (1 + 0.5 * module_mask)
        
        # 抑制模块间连接
        W = W * (1 - 0.4 * (1 - module_mask))

        # 添加稀疏化处理
        threshold = 0.4  # 设置阈值
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
        """执行完整的去个性化流程"""
        # 1. 预处理
        W1 = self.preprocessor.preprocess(W1)
        W2 = self.preprocessor.preprocess(W2)
        module_mask = self.preprocessor.initialize_module_structure()
        
        # 2. NMF分解
        H1 = self.nmf.decompose(W1, module_mask)
        H2 = self.nmf.decompose(W2, module_mask)
        
        # 3. 基底分析
        patterns = self.analyzer.analyze_patterns(H1)
        matches = self.analyzer.align_bases(H1, H2)
        
        # 4. 重构
        W_common = self.reconstructor.fuse_and_reconstruct(H1, H2, matches)
        W_common = self.reconstructor.optimize_structure(W_common, module_mask)
        
        # 5. 验证
        metrics = self.validator.evaluate_results(W1, W2, W_common, module_mask)
        
        return W_common, metrics, patterns

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
    
    # 为节点添加区域和颜色属性
    for i, name in enumerate(electrode_names):
        for region, electrodes in regions_mapping.items():
            if name in electrodes:
                G.nodes[i]['region'] = region
                G.nodes[i]['color'] = colors[region]
                G.nodes[i]['name'] = name
                break
    
    return G


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
            
            # if dist < 1e-6:
            #     continue
                
            mid_point = (start + end) / 2
            # Create perpendicular vector by rotating diff 90 degrees
            perp = np.array([-diff[1], diff[0]])
            
            # Normalize perpendicular vector
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 1e-6:  # Check for non-zero norm
                perp = perp / perp_norm
                
                # Calculate curvature based on distance
                curvature = 0.2 + 0.3 * (dist/2)
                
                # Calculate control point
                control_point = mid_point + curvature * perp
                
                # Create curved path
                verts = [tuple(start), tuple(control_point), tuple(end)]
                codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                path = Path(verts, codes)
                
                # Draw connection
                color = G.nodes[u]['color']
                alpha = min(abs(w['weight'])*3000, 0.8)
                line_width = abs(w['weight'])*5000
                patch = patches.PathPatch(path, facecolor='none', edgecolor=color,
                                        alpha=alpha, linewidth=line_width)
                ax.add_patch(patch)
    
    # Draw nodes
    for i in G.nodes():
        ax.plot(pos[i][0], pos[i][1], 'o', color=G.nodes[i]['color'],
                markersize=6, zorder=3)
        
        # Add labels with improved positioning
        angle = angles[i]
        label_radius = radius * 1.2
        label_pos = (label_radius * np.cos(angle), label_radius * np.sin(angle))
        
        # Adjust text alignment based on position
        ha = 'left' if -np.pi/2 <= angle <= np.pi/2 else 'right'
        va = 'center'
        
        # Rotate labels for better readability
        rotation = np.degrees(angle)
        if ha == 'right':
            rotation += 180
        if rotation > 90 and rotation <= 270:
            rotation -= 180
            
        plt.text(label_pos[0], label_pos[1], G.nodes[i]['name'],
                ha=ha, va=va, rotation=rotation,
                rotation_mode='anchor', fontsize=8)
    
    # Set plot properties
    plt.axis('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    plt.title('Brain Region Connectogram', pad=20)
    
    return plt


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


def get_model(args, edge_wight, edge_idx):
    return DGCNN(
        device=torch.device('cuda' if not args.cpu else 'cpu'),
        num_nodes=args.num_nodes,
        edge_weight=edge_wight,
        edge_idx=edge_idx,
        num_features=args.num_features,
        num_classes=args.num_classes,
        num_hiddens=args.num_hiddens,
        num_layers=args.num_layers,
    )


def main(args):
    # device = torch.device('cuda' if not args.cpu else 'cpu')
    # print("="*50)
    # print("训练正则化图")
    # group1, group2 = None, None
    # if args.group_mod == 'gender':
    #     group1, group2 = get_data_loaders_gender(args)
    # else:
    #     # todo  random
    #     raise ValueError("不支持的分组方法")
    # group1_trainer = Trainer.get_trainer(args)
    # group2_trainer = Trainer.get_trainer(args)
    # group1_trainer._set_data_loader(group1)
    # group2_trainer._set_data_loader(group2)
    # model_group1 = get_model(args, group1_trainer.edge_weight, group1_trainer.edge_index).to(device)
    # model_group2 = get_model(args, group2_trainer.edge_weight, group2_trainer.edge_index).to(device)
    # group1_trainer._set_model(model_group1)
    # group2_trainer._set_model(model_group2)
    # group1_trainer.init_optimizer()
    # group2_trainer.init_optimizer()
    # for i in range(args.num_epochs):
    #     print(f"Epoch {i}")
    #     group1_epoch_metrics = group1_trainer._train_with_eeg(args, i)
    #     group2_epoch_metrics = group2_trainer._train_with_eeg(args, i)
    #     print(f"Group1: {group1_epoch_metrics}")
    #     print(f"Group2: {group2_epoch_metrics}")
    # print("="*50)
    # trained_model1 = group1_trainer.get_model()
    # trained_model2 = group2_trainer.get_model()

    # # 热力图可视化
    # # visualize_lower_triangle(trained_model1.edge_weight.detach().cpu().numpy())
    # # visualize_lower_triangle(trained_model2.edge_weight.detach().cpu().numpy())

    # # 环形图可视化
    # G = matrix_to_connectogram_data(trained_model1.edge_weight.detach().cpu().numpy(), total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()
    # G = matrix_to_connectogram_data(trained_model2.edge_weight.detach().cpu().numpy(), total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()

    # # 保存edge weight
    # save_path = 'results'
    # # 以当前时间为文件名后缀
    # import time
    # import os
    # save_path = 'results'  # 或者你想要的其他路径
    # os.makedirs(save_path, exist_ok=True)
    # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # np.save(f'{save_path}/model_1_edge_path_{current_time}', trained_model1.edge_weight.detach().cpu().numpy())
    # np.save(f'{save_path}/model_2_edge_path_{current_time}', trained_model2.edge_weight.detach().cpu().numpy())





    group1_graphs = np.load('results/model_1_edge_path_2024-11-07-21-38-22.npy')
    group2_graphs = np.load('results/model_2_edge_path_2024-11-07-21-38-22.npy')
    group1_graphs = trans_triangular_to_full_matrix(group1_graphs)
    group2_graphs = trans_triangular_to_full_matrix(group2_graphs)
    deidentifier = ComprehensiveDeidentification(n_components=10)
    W_common, metrics, patterns = deidentifier.perform_deidentification(group1_graphs, group2_graphs)
    # 打印结果
    print("Metrics:", metrics)
    print("Patterns:", patterns)
    W_common = W_common / 1000
    G = matrix_to_connectogram_data(group1_graphs, total_part, regions_mapping)
    plot_connectogram(G, threshold=0.0001)
    plt.show()
    G = matrix_to_connectogram_data(group2_graphs, total_part, regions_mapping)
    plot_connectogram(G, threshold=0.0001)
    plt.show()
    G = matrix_to_connectogram_data(W_common, total_part, regions_mapping)
    plot_connectogram(G, threshold=0.0001)
    plt.show()

if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()













    # if args.train_fold == 'all':
    #     fold_list = np.arange(0, args.n_folds)
    # else:
    #     fold_list = [int(args.train_fold)]
    # result = Results(args)
    # buc = []
    # # gm = GPUManager()
    # args.device_index = 0
    # for i in fold_list:
    #     args_new = copy.deepcopy(args)
    #     args_new.fold_list = [i]
    #     buc.append(main(args_new))
    # para_mean_result_dict = {}
    # if args.subjects_type == 'inter':
    #     for tup in buc:
    #         result.acc_fold_list[tup[0]] = tup[1]
    #         result.subjectsScore[tup[2]] = tup[3]
    # elif args.subjects_type == 'intra':
    #     for tup in buc:
    #         result.acc_fold_list[tup[0]] = tup[1]
    #         result.subjects_results[:, tup[2]] = tup[3]
    #         result.label_val[:, tup[2]] = tup[4]
    # for tup in buc:
    #     if len(para_mean_result_dict) == 0:
    #         para_mean_result_dict = tup[-1]
    #     else:
    #         for k, v in tup[-1].items():
    #             para_mean_result_dict[k]['now_best_acc_train'] += v['now_best_acc_train']
    #             para_mean_result_dict[k]['now_best_acc_val'] += v['now_best_acc_val']
    # for k in para_mean_result_dict.keys():
    #     para_mean_result_dict[k]['now_best_acc_train'] /= len(para_mean_result_dict)
    #     para_mean_result_dict[k]['now_best_acc_val'] /= len(para_mean_result_dict)
    # json.dump({
    #     "para_mean_result_dict": para_mean_result_dict
    # }, open(os.path.join(args.model_path, 'para_mean_result_dict.json'), 'w'))

    # Utils.print_res(args, result)
    # Utils.draw_res(args)

