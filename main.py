import torch
import Utils
import Utils.Config
<<<<<<< HEAD
from Datasets.DataloaderUtils import get_data_loaders_gender, get_data_loaders_random, get_data_loader_cnn
import models.trainer as Trainer
from models.DGCNN import DGCNN
from models.CNNVAE import CNNVAE
from models.Utils import *
import numpy as np
import matplotlib.pyplot as plt
from Datasets.Datasets import VRSicknessDataset


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
        dropout=args.eeg_dropout,
        hidden_channels=args.edge_hidden_size,
        node_learnable=args.node_learnable,
    )


def get_cnn_model():
    return CNNVAE(
        input_size=(32,32)
    )

def train_cnn_vae(device: torch.device, args):
    # print("=" * 50)
    # print("训练CNN-VAE")
    train_loader, test_loader = get_data_loader_cnn(args)
    trainer = Trainer.get_cnn_trainer(args)
    trainer._set_data_loader(train_loader)
    model = get_cnn_model().to(device)
    trainer._set_model(model)
    trainer.init_optimizer()
    for epoch in range(args.num_epochs_video):
        # print(f"Epoch {epoch}")
        metric = trainer._train_with_original(args, epoch)
        # print(f"Train: {metric}")
    
    tester = Trainer.get_cnn_trainer(args)
    tester._set_data_loader(test_loader)
    tester._set_model(get_cnn_model().to(device))
    metric = tester._test_with_video(args)
    # print(f"Test: {metric}")


def train_normalize_eeg_gnn(device: torch.device, args):
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=['eeg'])
    best_acc = 0
    best_combo = None
    if not args.search:
        if args.group_mod == 'gender':
            group1, group2 = get_data_loaders_gender(args)
        else:
            train_loaders = get_data_loaders_random(args, datasets)
            group1, group2 = train_loaders[:2], train_loaders[2:]
        def setup_trainer(loader, args):
            trainer = Trainer.get_gnn_trainer(args)
            trainer._set_data_loader(loader)
            model = get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device)
            trainer._set_model(model)
            trainer.init_optimizer()
            return trainer

        trainers = [setup_trainer(loader, args) for loader in [group1[0]]]
        
        for epoch in range(args.num_epochs_gnn):
            metrics = [trainer._train_with_eeg(args, epoch) for trainer in trainers]
            for i, metric in enumerate(metrics, 1):
                print(f"Group{i}: {metric}")
        print("=" * 50)

        test_metrics = []
        for i, (trainer, test_loader) in enumerate(zip(trainers, [group1[1], group2[1]]), 1):
            tester = Trainer.get_gnn_trainer(args)
            tester._set_data_loader(test_loader)
            tester._set_model(get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device))
            metric = tester._test_with_eeg(args)
            test_metrics.append(metric)
            print(f"Group{i} Test: {metric}")
    else:
        param_dict = args.init_range_gnn()
        print("=" * 50, "Searching")
        for combo in param_dict:
            
            print(f"Combo: {combo}")
            
            args.group = combo['group']
            args.node_learnable = combo['node_learnable']
            args.eeg_hidden_size = combo['eeg_hidden_size']
            args.eeg_dropout = combo['eeg_dropout']
            args.num_layers = combo['num_layers']
            args.batch_size = combo['batch_size']
            args.data_sampler_strategy = combo['data_sampler_strategy']
            args.optimizer = combo['optimizer']
            args.rand_seed = combo['rand_seed']
            args.num_epochs_gnn = combo['num_epochs_gnn']
            args.l1_reg = combo['l1_reg']
            args.l2_reg = combo['l2_reg']
            args.lr = combo['lr']
            
            train_loaders = get_data_loaders_random(args, datasets)
            group1, group2 = train_loaders[:2], train_loaders[2:]
            
            def setup_trainer(loader, args):
                trainer = Trainer.get_gnn_trainer(args)
                trainer._set_data_loader(loader)
                model = get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device)
                trainer._set_model(model)
                trainer.init_optimizer()
                return trainer
            
            trainers = [setup_trainer(loader, args) for loader in [group1[0]]]
            
            for epoch in range(args.num_epochs_gnn):
                [trainer._train_with_eeg(args, epoch) for trainer in trainers]

            test_metrics = []
            for i, (trainer, test_loader) in enumerate(zip(trainers, [group1[1], group2[1]]), 1):
                tester = Trainer.get_gnn_trainer(args)
                tester._set_data_loader(test_loader)
                tester._set_model(get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device))
                metric = tester._test_with_eeg(args)
                test_metrics.append(metric)
                print(f"acc: {metric}")
            if test_metrics[0]['acc'] > best_acc:
                best_acc = test_metrics[0]['acc']
                best_combo = combo
        print("=" * 50)
        print(f"Best Combo: {best_combo}, Best Acc: {best_acc}")
        


def main(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')
    train_normalize_eeg_gnn(device, args)
    # train_cnn_vae(device, args)
    


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
    # import time
    # import os
    # save_path = 'results'  
    # os.makedirs(save_path, exist_ok=True)
    # current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    # np.save(f'{save_path}/model_1_edge_path_{current_time}', trained_model1.edge_weight.detach().cpu().numpy())
    # np.save(f'{save_path}/model_2_edge_path_{current_time}', trained_model2.edge_weight.detach().cpu().numpy())
=======
from Datasets.DataloaderUtils import get_data_loader_cnn, get_data_loaders_eeg_group
import models.trainer as Trainer
from models.DGCNN import get_eeg_model
from models.CNNVAE import get_cnn_model
from models.Utils import *


def train(args):
    device = torch.device('cuda' if not args.cpu else 'cpu')
    print("=" * 50)

    def setup_trainer(loader, args):
        trainer = Trainer.get_trainer(args)
        trainer._set_data_loader(loader)
        model = get_eeg_model(args, trainer.edge_weight,
                              trainer.edge_index).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        return trainer

    if args.model_mod == 'cnn':
        train_loader, test_loader = get_data_loader_cnn(args)

        trainer = Trainer.get_trainer(args)
        trainer._set_data_loader(train_loader)
        model = get_cnn_model().to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        for epoch in range(args.num_epochs_video):
            print(f"Epoch {epoch}")
            metrics = trainer._train_with_video(args, epoch)
        print(f"Train: {metrics}")

        tester = Trainer.get_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(trainer.get_model())
        model = model.to(device)
        model.eval()
        tester._set_model(model)
        metric = tester._test_with_video(args)
        print(f"Test: {metric}")

    elif args.model_mod == 'eeg_group':
        train_loaders = get_data_loaders_eeg_group(args)
        group1, group2 = train_loaders[:2], train_loaders[2:]

        trainers = [setup_trainer(loader, args)
                    for loader in [group1[0], group2[0]]]

        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            metrics = [trainer._train_with_eeg(
                args, epoch) for trainer in trainers]
            for i, metric in enumerate(metrics, 1):
                print(f"Group{i}: {metric}")
        print("=" * 50)

        test_metrics = []
        for i, (trainer, test_loader) in enumerate(zip(trainers, [group1[1], group2[1]]), 1):
            tester = Trainer.get_trainer(args)
            tester._set_data_loader(test_loader)
            tester._set_model(trainer.get_model())
            torch.save(trainer.get_model().state_dict(), f"{
                       args.model_save_path}/model_group_{i}.pth")
            metric = tester._test_with_eeg(args)
            test_metrics.append(metric)
            print(f"Group{i} Test: {metric}")

    elif args.mod == 'eeg':
        pass


def plot_ring_graph(G, threshold):
    """
    random seed = 42
    plot_ring_graph(G1, 0.0068)
    plot_ring_graph(G2, 0.009)
    plot_ring_graph(g_common, 0.0052)
    """
    G = (G - np.min(G)) / (np.max(G) - np.min(G))
    plt = visualize_brain_regions(G, threshold)
    plt.show()
>>>>>>> vim_branch


def depersonalization(args):
    G1_state_dict = torch.load(
        f"{args.model_save_path}/model_group_1.pth", weights_only=True)
    G2_state_dict = torch.load(
        f"{args.model_save_path}/model_group_2.pth", weights_only=True)
    G1_half = G1_state_dict['edge_weight'].detach().cpu().numpy()
    G2_half = G2_state_dict['edge_weight'].detach().cpu().numpy()
    G1_node = G1_state_dict['node_embedding'].detach().cpu().numpy()
    G2_node = G2_state_dict['node_embedding'].detach().cpu().numpy()

    G1 = trans_triangular_to_full_matrix(G1_half)
    G2 = trans_triangular_to_full_matrix(G2_half)

    common = CommonExtraction(G1, G2)
    common._get_common_base()
    common._get_common_activation()
    common._identify_common_pattern()
    g_common = common._get_common_matrix()
    node_common = (G1_node + G2_node) * 0.5
    return g_common, node_common


<<<<<<< HEAD
# # 
#     group1_graphs = np.load('results/model_2_edge_path_2024-11-07-21-38-22.npy')
#     group2_graphs = np.load('results/model_2_edge_path_2024-11-07-21-38-22.npy')
#     group1_graphs = trans_triangular_to_full_matrix(group1_graphs)
#     group2_graphs = trans_triangular_to_full_matrix(group2_graphs)
#     deidentifier = ComprehensiveDeidentification(n_components=16)
#     W_common, metrics, patterns = deidentifier.perform_deidentification(group1_graphs, group2_graphs)
    # G = matrix_to_connectogram_data(group1_graphs, total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()
    # G = matrix_to_connectogram_data(group2_graphs, total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()
    # G = matrix_to_connectogram_data(W_common, total_part, regions_mapping)
    # plot_connectogram(G, threshold=0.0001)
    # plt.show()
    
    
    # group1_graphs = group1_graphs * 100
    # region_matrix, region_names = matrix_to_region_matrix(group1_graphs, total_part, regions_mapping)
    # plot_region_connectogram(region_matrix, region_names, threshold=0.0001)
    # plt.show()
    
    
    # group2_graphs = group2_graphs * 100
    # region_matrix, region_names = matrix_to_region_matrix(group2_graphs, total_part, regions_mapping)
    # plot_region_connectogram(region_matrix, region_names, threshold=0.0001)
    # plt.show()
    
    # W_common = W_common / 20
    # region_matrix, region_names = matrix_to_region_matrix(W_common, total_part, regions_mapping)
    # plot_region_connectogram(region_matrix, region_names, threshold=0.0001)
    # plt.show()
    
    # group1_graphs = (group1_graphs - group1_graphs.min()) / (group1_graphs.max() - group1_graphs.min())
    # group2_graphs = (group2_graphs - group2_graphs.min()) / (group2_graphs.max() - group2_graphs.min())
    # W_common = (W_common - W_common.min()) / (W_common.max() - W_common.min())
    # region_matrix, region_names = matrix_to_region_matrix(W_common, total_part, regions_mapping)
    # # fig = plot_brain_connectivity_multiple_views(region_matrix, region_names, threshold=0.01)
    # # plt.show()
    # region_matrix, region_names = matrix_to_region_matrix(group1_graphs, total_part, regions_mapping)
    # fig = plot_brain_connectivity_multiple_views(region_matrix, region_names, threshold=0.001)
    # plt.show()
    # region_matrix, region_names = matrix_to_region_matrix(group2_graphs, total_part, regions_mapping)
    # fig = plot_brain_connectivity_multiple_views(region_matrix, region_names, threshold=0.001)
    # plt.show()
# 
    
=======
def main(args):
    # train(args)
    edge, node = depersonalization(args)

>>>>>>> vim_branch

if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()
