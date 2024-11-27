import torch
import Utils
import Utils.Config
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
        model = get_eeg_model(args, trainer.edge_weight, trainer.edge_index).to(device)
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

        trainers = [setup_trainer(loader, args) for loader in [group1[0], group2[0]]]
        
        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            metrics = [trainer._train_with_eeg(args, epoch) for trainer in trainers]
            for i, metric in enumerate(metrics, 1):
                print(f"Group{i}: {metric}")
        print("=" * 50)

        test_metrics = []
        for i, (trainer, test_loader) in enumerate(zip(trainers, [group1[1], group2[1]]), 1):
            tester = Trainer.get_trainer(args)
            tester._set_data_loader(test_loader)
            tester._set_model(trainer.get_model())
            torch.save(trainer.get_model().state_dict(), f"{args.model_save_path}/model_group_{i}.pth")
            metric = tester._test_with_eeg(args)
            test_metrics.append(metric)
            print(f"Group{i} Test: {metric}")
    
    elif args.mod == 'eeg':
        pass



def plot_ring_graph(G, threshold):
    G = (G - np.min(G))/ (np.max(G) - np.min(G))
    plt = visualize_brain_regions(G, threshold)
    plt.show() 

def depersonalization(args):
    G1_state_dict = torch.load(f"{args.model_save_path}/model_group_1.pth", weights_only=True)
    G2_state_dict = torch.load(f"{args.model_save_path}/model_group_2.pth", weights_only=True)
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
    
    
    """
    random seed = 42
    """
    plot_ring_graph(G1, 0.0068)
    plot_ring_graph(G2, 0.009)
    plot_ring_graph(g_common, 0.0052)


def main(args):
    # train(args)
    depersonalization(args)
    


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
    # save_path = 'results'
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