import torch
import Utils
import Utils.Config
from Datasets.DataloaderUtils import get_data_loaders_gender, get_data_loaders_random, get_data_loader_cnn
import models.trainer as Trainer
from models.DGCNN import DGCNN
from models.CNNVAE import CNNVAE

def get_gnn_model(args, edge_wight, edge_idx):
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


def get_cnn_model():
    return CNNVAE(
        input_size=(32,32)
    )

def train_cnn_vae(device: torch.device, args):
    print("=" * 50)
    print("训练CNN-VAE")
    train_loader, test_loader = get_data_loader_cnn(args)
    trainer = Trainer.get_cnn_trainer(args)
    trainer._set_data_loader(train_loader)
    model = get_cnn_model().to(device)
    trainer._set_model(model)
    trainer.init_optimizer()
    for epoch in range(args.num_epochs_video):
        print(f"Epoch {epoch}")
        metric = trainer._train_with_original(args, epoch)
        print(f"Train: {metric}")
    
    tester = Trainer.get_trainer(args)
    tester._set_data_loader(test_loader)
    tester._set_model(trainer.get_model())
    metric = tester._test_with_video(args)
    print(f"Test: {metric}")


def train_normalize_eeg_gnn(device: torch.device, args):
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
        trainer = Trainer.get_gnn_trainer(args)
        trainer._set_data_loader(loader)
        model = get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        return trainer

    trainers = [setup_trainer(loader, args) for loader in [group1[0], group2[0]]]
    
    # 训练循环
    for epoch in range(args.num_epochs_gnn):
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


def main(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')
    # train_normalize_eeg_gnn(device, args)
    train_cnn_vae(device, args)
    


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