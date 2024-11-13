import torch
import Utils
import Utils.Config
from Datasets.DataloaderUtils import get_data_loaders_gender, get_data_loaders_random
import models.trainer as Trainer
from models.DGCNN import DGCNN
import numpy as np
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
    device = torch.device('cuda' if not args.cpu else 'cpu')
    print("=" * 50)
    print("训练正则化图")

    # 获取数据加载器
    if args.group_mod == 'gender':
        group1, group2 = get_data_loaders_gender(args)
    else:
        train_loaders = get_data_loaders_random(args)
        group1, group2 = train_loaders[:2], train_loaders[2:]

    # 创建和训练模型
    def setup_trainer(loader, args):
        trainer = Trainer.get_trainer(args)
        trainer._set_data_loader(loader)
        model = get_model(args, trainer.edge_weight, trainer.edge_index).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        return trainer

    trainers = [setup_trainer(loader, args) for loader in [group1[0], group2[0]]]
    
    # 训练循环
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")
        metrics = [trainer._train_with_eeg(args, epoch) for trainer in trainers]
        for i, metric in enumerate(metrics, 1):
            print(f"Group{i}: {metric}")
    print("=" * 50)

    # 测试阶段
    test_metrics = []
    for i, (trainer, test_loader) in enumerate(zip(trainers, [group1[1], group2[1]]), 1):
        tester = Trainer.get_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(trainer.get_model())
        metric = tester._test_with_eeg(args)
        test_metrics.append(metric)
        print(f"Group{i} Test: {metric}")


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





    # group1_graphs = np.load('results/model_1_edge_path_2024-11-07-21-38-22.npy')
    # group2_graphs = np.load('results/model_2_edge_path_2024-11-07-21-38-22.npy')
    # group1_graphs = trans_triangular_to_full_matrix(group1_graphs)
    # group2_graphs = trans_triangular_to_full_matrix(group2_graphs)
    # deidentifier = ComprehensiveDeidentification(n_components=10)
    # W_common, metrics, patterns = deidentifier.perform_deidentification(group1_graphs, group2_graphs)
    # # 打印结果
    # print("Metrics:", metrics)
    # print("Patterns:", patterns)
    # W_common = W_common / 1000
    # G = matrix_to_connectogram_data(group1_graphs, total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()
    # G = matrix_to_connectogram_data(group2_graphs, total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()
    # G = matrix_to_connectogram_data(W_common, total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()

if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()