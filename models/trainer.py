import torch
import models.Utils as mUtils
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# TODO: improve train_xx and test_xx have a lot of similar code

def l1_reg_loss(model, only=None, exclude=None):
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


def l2_reg_loss(model, only=None, exclude=None):
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
    def __init__(self,
                batch_size=256, num_epoch=50, lr=0.005, dropout=0.5, early_stop=20, num_classes=2,
                optimizer='Adam', device=torch.device('cpu'),
                extension: dict = {}):
        # self.model: torch.nn.Module
        # self.data_loader: torch.utils.data.DataLoader
        self.num_epoch = num_epoch
        self.lr = lr
        self.optimizer_type = optimizer
        self.device = device
        self.early_stop = early_stop
        self.batch_size = batch_size
        if extension is not None:
            self.__dict__.update(extension)
        self.trainer_config_para = self.__dict__.copy()
        self.num_features = None
        self.model_config_para = dict()
        self.dropout = dropout
        self.num_classes = num_classes

    def _set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def _set_model(self, model):
        self.model = model

    def init_optimizer(self):
        if self.optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr)

    def get_model(self):
        self.model = self.model.eval()
        return self.model


class CNNTrainer(Trainer):
    def __init__(self, num_classes, device, dropout, batch_size, lr, num_epochs):
        super(CNNTrainer, self).__init__(
            num_classes=num_classes,
            batch_size=batch_size,
            num_epoch=num_epochs,
            lr=lr,
            dropout=dropout,
            device=device)

    def _train_with_video(self, args, epoch_num):
        self.model = self.model.train()
        epoch_loss = {
            'entropy_loss': 0.0,
        }
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []

        for batch in self.data_loader:
            self.optimizer.zero_grad()

            original_data = batch['original'].to(self.device, non_blocking=True)  \
                if 'original' in batch.keys() else None
            optical_data = batch['optical'].to(self.device, non_blocking=True) \
                if 'optical' in batch.keys() else None
            label = batch['label'].to(
                self.device, non_blocking=True).squeeze(1).long()
            output = self.model(original_data, optical_data)
            entropy_loss = F.cross_entropy(output, label)

            class_predict = output.argmax(axis=-1)
            class_predict = class_predict.cpu().detach().numpy()
            total_class_predictions += [item for item in class_predict]
            label = label.cpu().detach().numpy()
            num_correct_predict += np.sum(class_predict == label)

            loss = entropy_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=args.clip_norm)
            self.optimizer.step()

            with torch.no_grad():
                if original_data is not None:
                    total_samples += original_data.size(0)
                elif optical_data is not None:
                    total_samples += optical_data.size(0)
                else:
                    raise ValueError("original or optical data is required")
                epoch_loss['entropy_loss'] += entropy_loss.item()

        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = epoch_loss['entropy_loss'] / \
            (total_samples/self.batch_size)
        epoch_metrics['num_correct'] = num_correct_predict
        epoch_metrics['acc'] = num_correct_predict / total_samples

        return epoch_metrics

    def _test_with_video(self):
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []
        total_labels = []
        with torch.no_grad():
            for batch in self.data_loader:
                original_data = batch['original'].to(self.device, non_blocking=True)  \
                    if 'original' in batch.keys() else None
                optical_data = batch['optical'].to(self.device, non_blocking=True) \
                    if 'optical' in batch.keys() else None
                label = batch['label'].to(
                    self.device, non_blocking=True).squeeze(1).long()
                output = self.model(original_data, optical_data)
                class_predict = output.argmax(axis=-1)
                class_predict = class_predict.cpu().detach().numpy()
                total_class_predictions += [item for item in class_predict]
                label = label.cpu().detach().numpy()
                num_correct_predict += np.sum(class_predict == label)
                if original_data is not None:
                    total_samples += original_data.size(0)
                elif optical_data is not None:
                    total_samples += optical_data.size(0)
                else:
                    raise ValueError("original or optical data is required")
                total_labels += [item for item in label]

            y_true = np.array(total_labels)
            y_pred = np.array(total_class_predictions)

            epoch_metrics['num_correct'] = num_correct_predict
            epoch_metrics['acc'] = num_correct_predict / total_samples
            epoch_metrics['f1_macro'] = f1_score(
                y_true, y_pred, average='macro')
            epoch_metrics['f1_weighted'] = f1_score(
                y_true, y_pred, average='weighted')
            epoch_metrics['precision_macro'] = precision_score(
                y_true, y_pred, average='macro')
            epoch_metrics['recall_macro'] = recall_score(
                y_true, y_pred, average='macro')
            conf_matrix = confusion_matrix(y_true, y_pred)
            num_classes = conf_matrix.shape[0]
            fprs = []
            for i in range(num_classes):
                fp = np.sum(conf_matrix[:, i]) - \
                    conf_matrix[i, i]  # False Positives
                tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(
                    conf_matrix[:, i]) + conf_matrix[i, i]  # True Negatives
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fprs.append(fpr)

            epoch_metrics['false_positive_rates'] = fprs
            epoch_metrics['confusion_matrix'] = conf_matrix.tolist()

        return epoch_metrics


class DGCNNTrainer(Trainer):
    def __init__(self, edge_index, edge_weight,
                num_classes, device, num_hiddens,
                num_layers, dropout, batch_size,
                lr, l1_reg, l2_reg, num_epochs, optimizer):
        super(DGCNNTrainer, self).__init__(
            num_classes=num_classes,
            batch_size=batch_size,
            num_epoch=num_epochs,
            lr=lr,
            dropout=dropout,
            device=device,
            optimizer=optimizer)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.num_classes = num_classes
        self.device = device
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.lr = lr
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.num_epochs = num_epochs
        num_nodes = 0
        if edge_weight is not None:
            num_nodes = edge_weight.shape[0]
        self.num_nodes = num_nodes

        super(DGCNNTrainer, self).__init__(
            num_classes=num_classes,
            batch_size=batch_size,
            num_epoch=num_epochs,
            lr=lr,
            dropout=dropout,
            device=device)

    def _test_with_eeg(self):
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []
        overlap = []
        total_labels = []

        with torch.no_grad():
            for batch in self.data_loader:
                eeg_data = batch['eeg'].to(self.device, non_blocking=True)
                label = batch['label'].to(
                    self.device, non_blocking=True).squeeze(1).long()
                output = self.model(eeg_data)
                class_predict = output.argmax(axis=-1)
                class_predict = class_predict.cpu().detach().numpy()
                total_class_predictions += [item for item in class_predict]
                label = label.cpu().detach().numpy()
                num_correct_predict += np.sum(class_predict == label)
                total_samples += eeg_data.size(0)
                overlap.append([
                    f"{batch['sub_id'][i]}_{batch['slice_id'][i]}" 
                    for i in range(len(class_predict == label)) 
                    if (class_predict == label)[i]
                ])
                total_labels.extend([item for item in label])

            y_true = np.array(total_labels)
            y_pred = np.array(total_class_predictions)

            epoch_metrics['num_correct'] = num_correct_predict
            epoch_metrics['acc'] = num_correct_predict / total_samples
            epoch_metrics['f1_macro'] = f1_score(
                y_true, y_pred, average='macro')
            epoch_metrics['f1_weighted'] = f1_score(
                y_true, y_pred, average='weighted')
            epoch_metrics['precision_macro'] = precision_score(
                y_true, y_pred, average='macro')
            epoch_metrics['recall_macro'] = recall_score(
                y_true, y_pred, average='macro')

            conf_matrix = confusion_matrix(y_true, y_pred)
            num_classes = conf_matrix.shape[0]
            fprs = []
            for i in range(num_classes):
                fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]  # False Positives
                tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - \
                    np.sum(conf_matrix[:, i]) + conf_matrix[i, i]  # True Negatives
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fprs.append(fpr)

            epoch_metrics['false_positive_rates'] = fprs
            epoch_metrics['confusion_matrix'] = conf_matrix.tolist()
        return epoch_metrics

    def _train_with_eeg(self, args, epoch_num):
        self.model = self.model.train()
        epoch_loss = {
            'entropy_loss': 0.0,
            'l1_loss': 0.0,
            'l2_loss': 0.0
        }
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []

        for batch in self.data_loader:
            self.optimizer.zero_grad()

            eeg_data = batch['eeg'].to(self.device, non_blocking=True)
            label = batch['label'].to(
                self.device, non_blocking=True).squeeze(1).long()
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


def get_gnn_trainer(args) -> DGCNNTrainer:
    if args.cpu == True:
        device_using = torch.device('cpu')
    else:
        device_using = torch.device('cuda')
    _, edge_index, edge_weight = mUtils.get_edge_weight()
    optimizer = args.optimizer
    return DGCNNTrainer(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_classes=args.num_classes,
        device=device_using,
        num_hiddens=args.num_hiddens,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_reg=args.l1_reg,
        l2_reg=args.l2_reg,
        num_epochs=args.gnn_num_epochs,
        optimizer=optimizer
    )

# TODO: imporve this is issential part for get_trainer func

def _get_video_trainer():
    pass


class MCDISTrainer(Trainer):
    def __init__(self, device, dropout, batch_size, lr, num_epochs):
        super(MCDISTrainer, self).__init__(
            batch_size=batch_size,
            num_epoch=num_epochs,
            lr=lr,
            dropout=dropout,
            device=device)


    def _train(self, args, epoch_num):
        self.model.train()
        epoch_loss = {
            'total_loss': 0.0,
            'contrast_loss': 0.0,
            'class_loss': 0.0,
            'rebuild_loss': 0.0,
        }
        total_samples = 0
        num_correct_predict = 0

        for batch in self.data_loader:
            self.optimizer.zero_grad()
            eeg_data = batch['eeg'].to(self.device, non_blocking=True)
            original_data = batch['original'].to(self.device, non_blocking=True)
            optical_data = batch['optical'].to(self.device, non_blocking=True)
            motion_data = batch['motion'].to(self.device, non_blocking=True)
            label = batch['label'].to(self.device, non_blocking=True).squeeze(1).long()

            output, personal_graph, video_graph, depersonal_graph = self.model(
                eeg_data, original_data, optical_data, motion_data
            )

            _loss = self.model.loss(personal_graph, video_graph, depersonal_graph, output, label)
            total_loss = _loss['total_loss']
            contrast_loss = _loss['contrast_loss']
            class_loss = _loss['class_loss']
            rebuild_loss = _loss['rebuild_loss']

            # TODO: important where cal the reg items?
            # TODO: important different loss use different optimizer optimize different params
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args.clip_norm)
            self.optimizer.step()

            with torch.no_grad():
                batch_size = eeg_data.size(0)
                total_samples += batch_size
                epoch_loss['total_loss'] += total_loss.item() * batch_size
                epoch_loss['contrast_loss'] += contrast_loss.item() * batch_size
                epoch_loss['class_loss'] += class_loss.item() * batch_size
                epoch_loss['rebuild_loss'] += rebuild_loss.item() * batch_size

                pred = torch.argmax(output, dim=1)
                num_correct_predict += (pred == label).sum().item()

        epoch_loss = {k: v / total_samples for k, v in epoch_loss.items()}
        acc = num_correct_predict / total_samples if total_samples > 0 else 0.0

        epoch_metrics = {
            'epoch': epoch_num,
            'loss': list(epoch_loss.items()),
            'num_correct': num_correct_predict,
            'acc': acc
        }

        return epoch_metrics
    
    def _test(self):
        total_samples = 0
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []
        total_labels = []
        
        with torch.no_grad():
            for batch in self.data_loader:
                eeg_data = batch['eeg'].to(self.device, non_blocking=True)
                original_data = batch['original'].to(self.device, non_blocking=True)
                optical_data = batch['optical'].to(self.device, non_blocking=True)
                motion_data = batch['motion'].to(self.device, non_blocking=True)
                label = batch['label'].to(self.device, non_blocking=True).squeeze(1).long()

                output, _, _, _ = self.model(eeg_data, original_data, optical_data, motion_data)
                pred = torch.argmax(output, dim=1)
                num_correct_predict += (pred == label).sum().item()

                total_samples += eeg_data.size(0)
                
            epoch_metrics['num_correct'] = num_correct_predict
            epoch_metrics['acc'] = num_correct_predict / total_samples
            
            y_true = np.array(total_labels)
            y_pred = np.array(total_class_predictions)
            
            epoch_metrics['num_correct'] = num_correct_predict
            epoch_metrics['acc'] = num_correct_predict / total_samples
            epoch_metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
            epoch_metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
            epoch_metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
            epoch_metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
            
            conf_matrix = confusion_matrix(y_true, y_pred)
            num_classes = conf_matrix.shape[0]
            
            fprs = []
            for i in range(num_classes):
                fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
                tn = np.sum(conf_matrix) - np.sum(conf_matrix[i, :]) - np.sum(conf_matrix[:, i]) + conf_matrix[i, i]
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fprs.append(fpr)
                
            epoch_metrics['false_positive_rates'] = fprs
            epoch_metrics['confusion_matrix'] = conf_matrix.tolist()
        return epoch_metrics
        
def get_video_trainer(args) -> CNNTrainer:
    return CNNTrainer(
        num_classes=args.num_classes,
        device=torch.device('cuda' if not args.cpu else 'cpu'),
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs
    )

def get_eeg_trainer(args) -> DGCNNTrainer:
    _, edge_index, edge_weight = mUtils.get_edge_weight()
    return DGCNNTrainer(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_classes=args.num_classes,
        device=torch.device('cuda' if not args.cpu else 'cpu'),
        num_hiddens=args.num_hiddens,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        l1_reg=args.l1_reg,
        l2_reg=args.l2_reg,
        num_epochs=args.gnn_num_epochs,
        optimizer=args.optimizer
    )

def get_all_trainer(args) -> MCDISTrainer:
    return MCDISTrainer(
        device=torch.device('cuda' if not args.cpu else 'cpu'),
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs
    )