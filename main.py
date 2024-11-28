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
    """
    random seed = 42
    plot_ring_graph(G1, 0.0068)
    plot_ring_graph(G2, 0.009)
    plot_ring_graph(g_common, 0.0052)
    """
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
    node_common = (G1_node + G2_node) * 0.5
    return g_common, node_common
    
    


def main(args):
    # train(args)
    edge, node = depersonalization(args)
    
if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()