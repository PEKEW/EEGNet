import torch.nn as nn
import torch.utils.tensorboard
from models.Dtf import DTF
from models.CNNVAE import get_cnn_model
from models.DGCNN import get_gnn_model
from Datasets.DataloaderUtils import get_data_loader_cnn, \
    get_data_loaders_eeg_no_group, get_data_loader_all, get_data_loaders_eeg_group
from Datasets.Datasets import VRSicknessDataset
import numpy as np
from models.Utils import *
from models.All import MCDIS
import models.trainer as Trainer
import torch
import Utils.Config
from models.Utils import get_edge_weight
from typing import Tuple, Optional, Dict
from torch.utils.tensorboard.writer import SummaryWriter


# TODO: improve remove all print | gradient

# TODO: improve trans this func to models.All
def get_pretrained_info(args) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor]:
    """
    get pertrained edge expr and node expr
    edge expr: 30, 30
    node expr: 30, 250
    """
    def load(path):
        w = torch.load(path, weights_only=True)
        edge = nn.Parameter(w['edge_weight'].detach(), requires_grad=True)
        node = nn.Parameter(w['node_embedding'].detach(), requires_grad=True)
        idx = w.state_dict()['edge_idx']
        return edge, node, idx
    try:
        pretrain_edge_expr_1, pretrain_node_expr_1, _ = load(f"{args.model_save_path}/model_group_1.pth")
        pretrain_edge_expr_2, pretrain_node_expr_2, edge_idx = load(f"{args.model_save_path}/model_group_2.pth")
        pretrain_edge_expr = (pretrain_edge_expr_1 + pretrain_edge_expr_2) * 0.5
        pretrain_node_expr = (pretrain_node_expr_1 + pretrain_node_expr_2) * 0.5
    except Exception:
        print("fail to load pretrained model, use random initialization")
        _, edge_idx, pretrain_edge_expr = get_edge_weight()
        pretrain_edge_expr = nn.Parameter(torch.Tensor(
            pretrain_edge_expr).float(), requires_grad=True)
        pretrain_node_expr = nn.Parameter(
            torch.randn(30, 250).float(), requires_grad=True)
    if not pretrain_edge_expr.requires_grad:
        pretrain_edge_expr.requires_grad = True
    if not pretrain_node_expr.requires_grad:
        pretrain_node_expr.requires_grad = True
    device = torch.device('cuda' if not args.cpu else 'cpu')
    pretrain_edge_expr = pretrain_edge_expr.to(device)
    pretrain_node_expr = pretrain_node_expr.to(device)
    return pretrain_edge_expr, edge_idx, pretrain_node_expr


def get_dtf(args):
    dft = DTF(args)
    try:
        dft.load_state_dict(torch.load(args.dtf_path, weights_only=True))
    except Exception:
        print("warring: DTF not be trained yet or trained model not found")
    return dft


def get_mcdis_model(args):
    pretrain_edge_expr, edge_idx, pretrain_node_expr = get_pretrained_info(
        args)
    dtf = get_dtf(args)
    return MCDIS(
        args=args,
        edge_weight=pretrain_edge_expr,
        node_weight=pretrain_node_expr,
        edge_idx=edge_idx,
        dtf=dtf
    )


def train_cnn_vae(device: torch.device, args: Utils.Config.Args):
    train_loader, test_loader = get_data_loader_cnn(args)
    trainer: Trainer.CNNTrainer = Trainer.get_video_trainer(args)
    trainer._set_data_loader(train_loader)
    model = get_cnn_model().to(device)
    trainer._set_model(model)
    trainer.init_optimizer()
    for epoch in range(args.cnn_num_epochs):
        # print(f"Epoch {epoch}")
        metric = trainer._train_with_video(args, epoch)
    tester = Trainer.get_video_trainer(args)
    tester._set_data_loader(test_loader)
        # print(f"Train: {metric}")

    tester._set_model(get_cnn_model().to(device))
    metric = tester._test_with_video()
    print(f"Test: {metric}")


def setup_and_train_model(group1, group2, args, device):
    def setup_trainer(loader, args):
        trainer = Trainer.get_gnn_trainer(args)
        trainer._set_data_loader(loader)
        model = get_gnn_model(args, trainer.edge_weight,
                            trainer.edge_index).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        return trainer

    trainers = [setup_trainer(loader, args) for loader in [group1[0]]]

    # Training
    for epoch in range(args.gnn_num_epochs):
        metrics = [trainer._train_with_eeg(
            args, epoch) for trainer in trainers]

    test_metrics = []
    for trainer, test_loader in zip(trainers, [group1[1], group2[1]]):
        tester = Trainer.get_gnn_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(get_gnn_model(
            args, trainer.edge_weight, trainer.edge_index).to(device))
        metric = tester._test_with_eeg()
        test_metrics.append(metric)

    return test_metrics


def update_model_paras(args, params):
    raise NotImplementedError


def train_normalize_eeg_gnn(device: torch.device, args: Utils.Config.Args) -> Tuple[float, Optional[Dict]]:
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=['eeg'])
    best_accuracy = 0
    best_parameters = None

    if not args.search:
        train_loaders = get_data_loaders_eeg_no_group(args)
        group1, group2 = train_loaders[:2], train_loaders[2:]
        test_metrics = setup_and_train_model(group1, group2, args, device)

        for i, metric in enumerate(test_metrics, 1):
            print(f"Group{i} Test: {metric}")
    else:
        parameter_combinations = args.init_range_gnn()
        print("=" * 50, "Searching")
        for params in parameter_combinations:
            print(f"Testing parameters: {params}")
            update_model_paras(args, params)
            try:
                train_loaders = get_data_loaders_eeg_no_group(args)
                group1, group2 = train_loaders[:2], train_loaders[2:]
                test_metrics = setup_and_train_model(
                    group1, group2, args, device)

                if test_metrics[0]['accuracy'] > best_accuracy:
                    best_accuracy = test_metrics[0]['accuracy']
                    best_parameters = params

                print(f"Accuracy: {test_metrics[0]['accuracy']}")
            except Exception as e:
                print(f"Error during training with parameters {params}: {str(e)}")
                continue

        print("=" * 50)
        print(f"Best Parameters: {best_parameters}, Best Accuracy: {best_accuracy}")

    return best_accuracy, best_parameters


def train(args, writer):
    device = torch.device('cuda' if not args.cpu else 'cpu')
    print("=" * 50)

    def setup_trainer(loader, args):
        trainer = Trainer.get_eeg_trainer(args)
        trainer._set_data_loader(loader)
        model = get_gnn_model(args, trainer.edge_weight,
                            trainer.edge_index).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        return trainer

    # TODO: improve: trans these code to funcs
    if args.model_mod == 'cnn':
        train_loader, test_loader = get_data_loader_cnn(args)

        trainer = Trainer.get_video_trainer(args)
        trainer._set_data_loader(train_loader)
        model = get_cnn_model().to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        for epoch in range(args.cnn_num_epochs):
            print(f"Epoch {epoch}")
            metrics = trainer._train_with_video(args, epoch)
            print(f"Train: {metrics}")

        tester = Trainer.get_video_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(trainer.get_model())
        metric = tester._test_with_video()
        print(f"Test: {metric}")

    elif args.model_mod == 'eeg_group':
        train_loaders = get_data_loaders_eeg_no_group(args)
        group1, group2 = train_loaders[:2], train_loaders[2:]

        trainers = [setup_trainer(loader, args)
                    for loader in [group1[0], group2[0]]
                    ]

        for epoch in range(args.num_epochs):
            print(f"Epoch {epoch}")
            metrics = [trainer._train_with_eeg(
                args, epoch) for trainer in trainers]
            for i, metric in enumerate(metrics, 1):
                print(f"Group{i}: {metric}")
        print("=" * 50)

        test_metrics = []
        for i, (trainer, test_loader) in \
                enumerate(zip(trainers, [group1[1], group2[1]]), 1):
            tester = Trainer.get_eeg_trainer(args)
            tester._set_data_loader(test_loader)
            tester._set_model(trainer.get_model())
            torch.save(trainer.get_model().state_dict(), f"{args.model_save_path}/model_group_{i}.pth")
            metric = tester._test_with_eeg()
            test_metrics.append(metric)
            print(f"Group{i} Test: {metric}")

    elif args.model_mod == 'eeg':
        raise NotImplementedError("Not implemented yet")
    elif args.model_mod == 'all':
        train_loader, test_loader = get_data_loader_all(args)
        trainer = Trainer.get_all_trainer(args)
        trainer._set_data_loader(train_loader)
        model = get_mcdis_model(args).to(device)
        trainer._set_model(model)
        trainer.init_optimizer(mulit_optimizer=True)
        tester = Trainer.get_all_trainer(args)
        tester._set_data_loader(test_loader)
        for epoch in range(args.mcdis_num_epochs):
            print(f"Epoch {epoch}")
            metrics = trainer._train(args, epoch, writer)
            writer.add_scalars('Loss/train', {
                'ALoss': metrics['loss'][0][1],
                'CoLoss': metrics['loss'][1][1],
                'ClLoss': metrics['loss'][2][1],
                'RLoss': metrics['loss'][3][1],
            }, epoch)
            print(f"Train: {metrics}")
            writer.add_scalars('Acc', 
                {"train":metrics['acc'], }, epoch)
            # writer.add_scalars('Acc', 
            #     {"train":metrics['acc'], 
            #     "test": metric['acc']}, epoch)
        print("=" * 50)
        tester._set_model(trainer.get_model())
        metric = tester._test()
        print(f"Test: {metric}")



# TODO: improve trans these func to Utils
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
    common._identify_common_pattern(0.02)
    g_common = common._get_common_base()
    node_common = (G1_node + G2_node) * 0.5
    return g_common, node_common


def main(args):
    writer = SummaryWriter(log_dir = args.dash_path)
    train(args, writer)
    # edge, node = depersonalization(args)
    # train_normalize_eeg_gnn(device, args)


if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()
