import torch
import models.Utils as mUtils
import numpy as np
from torch.nn import functional as F

def l1_reg_loss(model,only=None,exclude=None): 
    total_loss = 0
    if only is None and exclude is None:
        for name, param in model.named_parameters():
            total_loss += torch.sum(torch.abs(param))
    elif only is not None:
        for name, param in model.named_parameters():
            if name in only:
                total_loss += torch.sum(torch.abs(param))
    elif exclude is not None:
        for name, param in model.named_parameters():
            if name not in exclude:
                total_loss += torch.sum(torch.abs(param))
    return total_loss

def l2_reg_loss(model,only=None,exclude=None): 
    total_loss = 0
    if only is None and exclude is None:
        for name, param in model.named_parameters():
            total_loss += torch.sum(torch.square(param))
    elif only is not None:
        for name, param in model.named_parameters():
            if name in only:
                total_loss += torch.sum(torch.square(param))
    elif exclude is not None:
        for name, param in model.named_parameters():
            if name not in exclude:
                total_loss += torch.sum(torch.square(param))
    return total_loss

class Trainer(object):
    def __init__(self, num_nodes, num_hiddens=400, num_classes=2,
                 batch_size=256, num_epoch=50, lr=0.005, l1_reg=0, l2_reg=0, dropout=0.5, early_stop=20,
                 optimizer='Adam', device=torch.device('cpu'),
                 extension: dict = None):
        self.num_nodes = num_nodes
        self.num_hiddens = num_hiddens
        self.num_epoch = num_epoch
        self.num_classes = num_classes
        self.dropout = dropout
        self.lr = lr
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.device = device
        self.early_stop = early_stop
        if extension is not None:
            self.__dict__.update(extension)
        self.trainer_config_para = self.__dict__.copy()
        self.model = None
        self.num_features = None
        self.model_config_para = dict()
        self.data_loader = None

    def _set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def _set_model(self, model):
        self.model = model


    def init_optimizer(self):
        self.optimizer = 'SGD'
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr)

    def save(self, path, name='best_model.dic.pkl'):
        raise NotImplementedError

    def load(self, path, name='best_model.dic.pkl'):
        raise NotImplementedError

class DGCNNTrainer(Trainer):
    def __init__(self, edge_index, edge_weight, num_classes, device, num_hiddens, num_layers, dropout, batch_size, lr, l1_reg, l2_reg, num_epochs):
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_classes = num_classes
        self.device = device
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.l2_reg = l2_reg
        self.num_epochs = num_epochs
        num_nodes = 0
        if edge_weight is not None:
            num_nodes = edge_weight.shape[0]
        super(DGCNNTrainer, self).__init__(
            num_nodes=num_nodes, 
            num_hiddens=num_hiddens,
            num_classes=num_classes,
            batch_size=batch_size,
            num_epoch=num_epochs,
            lr=lr,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            dropout=dropout,
            device=device)


    def _test_with_eeg(self, args):
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []

        with torch.no_grad():
            for batch in self.data_loader:
                eeg_data = batch['eeg'].to(self.device, non_blocking=True)
                label = batch['label'].to(self.device, non_blocking=True).squeeze(1).long()
                output = self.model(eeg_data)
                class_predict = output.argmax(axis=-1)
                class_predict = class_predict.cpu().detach().numpy()
                total_class_predictions += [item for item in class_predict]
                label = label.cpu().detach().numpy()
                num_correct_predict += np.sum(class_predict == label)
                total_samples += eeg_data.size(0)
            
            epoch_metrics['num_correct'] = num_correct_predict
            epoch_metrics['acc'] = num_correct_predict / total_samples
        return epoch_metrics


    def _train_with_eeg(self, args, epoch_num):
        self.model = self.model.train()
        epoch_loss = {
            'entropy_loss': 0,
            'l1_loss': 0,
            'l2_loss': 0
        }
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []

        for batch in self.data_loader:
            self.optimizer.zero_grad()

            eeg_data = batch['eeg'].to(self.device, non_blocking=True)
            label = batch['label'].to(self.device, non_blocking=True).squeeze(1).long()
            output = self.model(eeg_data)
            entropy_loss = F.cross_entropy(output, label)
            
            class_predict = output.argmax(axis=-1)
            class_predict = class_predict.cpu().detach().numpy()
            total_class_predictions += [item for item in class_predict]
            label = label.cpu().detach().numpy()
            num_correct_predict += np.sum(class_predict == label)

            l1_loss = self.l1_reg * l1_reg_loss(
                self.model, only=['edge_weight'])
            l2_loss = self.l2_reg * l2_reg_loss(
                self.model, exclude=['edge_weight'])
            loss = entropy_loss + l1_loss + l2_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=args.clip_norm)
            self.optimizer.step()

            with torch.no_grad():
                total_samples += eeg_data.size(0)
                epoch_loss['entropy_loss'] += entropy_loss.item()
                epoch_loss['l1_loss'] += l1_loss.item()
                epoch_loss['l2_loss'] += l2_loss.item()
        
        epoch_metrics['epoch'] = epoch_num
        
        epoch_metrics['loss'] = (
            epoch_loss['l2_loss'] + 
            epoch_loss['l1_loss'] +
            epoch_loss['entropy_loss']
    ) / (total_samples/self.batch_size)
        
        epoch_metrics['num_correct'] = num_correct_predict
        epoch_metrics['acc'] = num_correct_predict / total_samples

        return epoch_metrics
    
    def get_model(self):
        self.model = self.model.eval()
        return self.model


def get_trainer(args) -> DGCNNTrainer:
    if args.cpu == True:
        deviceUsing = torch.device('cpu')
    else:
        deviceUsing = torch.device('cuda')
    _, edge_index, edge_weight = mUtils.get_edge_weight()
    return DGCNNTrainer(
            edge_index = edge_index,
            edge_weight = edge_weight,
            num_classes=args.num_classes,
            device=deviceUsing,
            num_hiddens = args.num_hiddens,
            num_layers = args.num_layers,
            dropout = args.dropout,
            batch_size = args.batch_size,
            lr = args.lr,
            l1_reg = args.l1_reg,
            l2_reg = args.l2_reg,
            num_epochs = args.num_epochs
        )