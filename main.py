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


# TODO: improve trans this func to models.All
def _get_edge_weight(args):
    try:
        w = torch.load(f"{args.model_save_path}/model_group_1.pth")
        edge_weight = w.state_dict()['edge_weight']
        edge_idx = w.state_dict()['edge_idx']
    except Exception:
        print("fail to load pretrained model")
        _, edge_weight, edge_idx = get_edge_weight()
    return edge_weight, edge_idx


def get_dtf(args):
    # TODO: important return diffusion module
    from models.Dtf import DTF
    dft = DTF(args)
    try:
        dft.load_state_dict(torch.load(args.dtf_path))
    except Exception:
        print("fail to load depersonal model")
    return dft


def get_mcdis_model(args):
    edge_weight, edge_idx = _get_edge_weight(args)
    return MCDIS(
        args=args,
        edge_weight=edge_weight,
        edge_idx=edge_idx,
        Dtf=get_dtf(args)
    )


def train_cnn_vae(device: torch.device, args: Utils.Config.Args):
    train_loader, test_loader = get_data_loader_cnn(args)
    trainer:Trainer.CNNTrainer = Trainer.get_video_trainer(args)
    trainer._set_data_loader(train_loader)
    model = get_cnn_model().to(device)
    trainer._set_model(model)
    trainer.init_optimizer()
    for epoch in range(args.num_epochs_video):
        # print(f"Epoch {epoch}")
        metric = trainer._train_with_video(args, epoch)
        # print(f"Train: {metric}")

    tester = Trainer.get_video_trainer(args)
    tester._set_data_loader(test_loader)
    tester._set_model(get_cnn_model().to(device))
    metric = tester._test_with_video(args)
    print(f"Test: {metric}")

def setup_and_train_model(group1, group2, args, device):
    def setup_trainer(loader, args):
        trainer = Trainer.get_gnn_trainer(args)
        trainer._set_data_loader(loader)
        model = get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        return trainer
    
    trainers = [setup_trainer(loader, args) for loader in [group1[0]]]
    
    # Training
    for epoch in range(args.num_epochs_gnn):
        metrics = [trainer._train_with_eeg(args, epoch) for trainer in trainers]
    
    # Testing
    test_metrics = []
    for trainer, test_loader in zip(trainers, [group1[1], group2[1]]):
        tester = Trainer.get_gnn_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(get_gnn_model(args, trainer.edge_weight, trainer.edge_index).to(device))
        metric = tester._test_with_eeg()
        test_metrics.append(metric)
    
    return test_metrics


def update_model_paras(args, params):
    # TODO: important is needs search params, fix this
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
        print("="* 50, "Searching")
        for params in parameter_combinations:
            print(f"Testing parameters: {params}")
            update_model_paras(args, params)
            try:
                train_loaders = get_data_loaders_eeg_no_group(args)
                group1, group2 = train_loaders[:2], train_loaders[2:]
                test_metrics = setup_and_train_model(group1, group2, args, device)
                
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


def train(args):
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
        for epoch in range(args.num_epochs_video):
            print(f"Epoch {epoch}")
            metrics = trainer._train_with_video(args, epoch)
            print(f"Train: {metrics}")

        tester = Trainer.get_video_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(trainer.get_model())
        metric = tester._test_with_video(args)
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
            torch.save(trainer.get_model().state_dict(), f"{
                    args.model_save_path}/model_group_{i}.pth")
            metric = tester._test_with_eeg()
            test_metrics.append(metric)
            print(f"Group{i} Test: {metric}")

    elif args.model_mod == 'eeg':
        raise NotImplementedError("Not implemented yet")
    elif args.model_mod == 'all':
        train_loader, test_loader = get_data_loader_all(args)
        trainer = Trainer.get_all_trainer(args)
        model = get_mcdis_model(args).to(device)
        trainer._set_model(model)
        trainer.init_optimizer()
        for epoch in range(args.num_epoch_mcdis):
            print(f"Epoch {epoch}")
            metrics = trainer._train(args, epoch)
            print(f"Train: {metrics}")
        print("=" * 50)

        tester = Trainer.get_all_trainer(args)
        tester._set_data_loader(test_loader)
        tester._set_model(trainer.get_model())
        metric = tester._test(args)
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
    train(args)
    # edge, node = depersonalization(args)
    # train_normalize_eeg_gnn(device, args)


if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()
