from typing import List
import torch
import models.Utils as mUtils
import math
import numpy as np
import os
import time
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from models.DGCNN import DGCNN
import copy

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


    def init_optimizer(self, model):
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(
                model.parameters(), lr=self.lr)

    def do_epoch(self, data_loader, mode='eval', epoch_num=None, ret_predict=False):
        if mode == 'train':
            self.model = self.model.train()
        else:
            self.model = self.model.eval()
        epoch_loss = {'Entropy_loss': 0, 'L1_loss': 0, 'L2_loss': 0,
                      'NodeDAT_loss': 0} 
        total_samples = 0  
        epoch_metrics = {}
        num_correct_predict = 0
        total_class_predictions = []

        for _, batch in enumerate(data_loader):
            X, Y = batch
            num_samples = X.shape[0]
            predictions = self.model(X)

            Entropy_loss = self.loss_module(
                predictions.float(),
                torch.Tensor(Y.float()).long().to(self.device))

            class_predict = predictions.argmax(axis=-1)
            class_predict = class_predict.cpu().detach().numpy()
            if ret_predict == True:
                total_class_predictions += [
                    item for item in class_predict]

            Y = Y.cpu().detach().numpy()
            num_correct_predict += np.sum(class_predict == Y)
            L1_loss = self.l1_reg * l1_reg_loss(
                self.model, only=['edge_weight']) 
            L2_loss = self.l2_reg * l2_reg_loss(
                self.model, exclude=['edge_weight'])

            loss = Entropy_loss + L1_loss + L2_loss


            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=20)  # change to 20
                self.optimizer.step()
            
            with torch.no_grad():
                for name, para in self.model.named_parameters():
                    if name in ['edge_weight']:
                        tmp = F.relu(para.data)
                        para.copy_(tmp)

            with torch.no_grad():
                total_samples += num_samples
                epoch_loss['Entropy_loss'] += Entropy_loss.item()
                epoch_loss['L1_loss'] += L1_loss  # .item()
                epoch_loss['L2_loss'] += L2_loss  # .item()

        epoch_metrics['epoch'] = epoch_num
        epoch_metrics['loss'] = (
            epoch_loss['L2_loss']+epoch_loss['L1_loss']+epoch_loss['Entropy_loss']).item() / (total_samples/self.batch_size)
        epoch_metrics['num_correct'] = num_correct_predict
        epoch_metrics['acc'] = num_correct_predict/total_samples

        if ret_predict == False:
            return epoch_metrics
        else:
            return total_class_predictions

    def train_and_eval(self, train_data, train_label, valid_data, valid_label,
                       num_freq=None, reload=True, nd_predict=True,train_log=False): 
        # todo train data -> train loader
        self.num_features = train_data.shape[-1]
        self.trainer_config_para['num_features'] = self.num_features
        for k, v in self.__dict__.items():
            if k in self.model_config_para_name:
                self.model_config_para.update({k: v})
        self.trainer_config_para['model_config_para'] = self.model_config_para

        self.model = self.model_f(**self.model_config_para)
        self.model.to(self.device)

        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr)

        train_loader, valid_loader = self.data_prepare(
            train_data, train_label, valid_data, valid_label)

        best_acc = 0
        early_stop = self.early_stop
        if early_stop is None:
            early_stop = self.num_epoch
        early_stop_num = 0
        eval_acc_list = []
        train_acc_list = []
        eval_loss_list = []
        train_num_correct_list = []
        eval_num_correct_list = []
        train_loss_list = []
        for i in range(self.num_epoch):
            time_start = time.time()
            train_metric = self.do_epoch(train_loader, 'train', i)
            eval_metric = self.do_epoch(valid_loader, 'eval', i)

            time_end = time.time()
            time_cost = time_end - time_start

            if train_log==True:
                print('device', self.device, 'Epoch {:.0f} training_acc: {:.4f}  valid_acc: {:.4f}| train_loss: {:.4f}, valid_loss: {:.4f}, | time cost: {:.3f}'.format(
                    train_metric['epoch'], train_metric['acc'], eval_metric['acc'], train_metric['loss'], eval_metric['loss'], time_cost))

            eval_acc_list.append(eval_metric['acc'])
            train_acc_list.append(train_metric['acc'])
            eval_loss_list.append(eval_metric['loss'])
            train_loss_list.append(train_metric['loss'])
            eval_num_correct_list.append(eval_metric['num_correct'])
            train_num_correct_list.append(train_metric['num_correct'])
            if eval_metric['acc'] > best_acc:
                early_stop_num = 0
                best_acc = eval_metric['acc']
            else:
                early_stop_num += 1


        self.eval_acc_list = eval_acc_list
        self.train_acc_list = train_acc_list
        

    def train_only(self, train_data, train_label, valid_data=None, mat_train=None,
                   num_freq=None,train_log=False):  # ,small=False,step=0.00001):
        # todo train data -> train loader

        self.num_features = train_data.shape[-1]
        self.trainer_config_para['num_features'] = self.num_features
        for k, v in self.__dict__.items():
            if k in self.model_config_para_name:
                self.model_config_para.update({k: v})
        self.trainer_config_para['model_config_para'] = self.model_config_para

        self.model = self.model_f(**self.model_config_para)
        self.model.to(self.device)

        train_loader = self.data_prepare_train_only(
            train_data, train_label)

        train_acc_list = []
        train_num_correct_list = []
        train_loss_list = []
        for i in range(self.num_epoch):
            if train_log:
                print("training epochs : ", i)
            train_metric = self.do_epoch(train_loader, 'train', i)
            # print('device',self.device,'fold', fold, 'Epoch {:.1f} training_acc: {:.4f}  valid_acc: {:.4f}| train_loss: {:.4f}, valid_loss: {:.4f}, | time cost: {:.3f}'.format(
            #     train_metric['epoch'], train_metric['acc'], eval_metric['acc'], train_metric['loss'], eval_metric['loss'], time_cost))
            train_acc_list.append(train_metric['acc'])
            train_loss_list.append(train_metric['loss'])
            train_num_correct_list.append(train_metric['num_correct'])

        self.train_acc_list = train_acc_list
        return train_acc_list

    def predict(self, data, adj_mat=None):  # inference
        if self.model is None:
            raise Exception(
                f"{self.model_name} model has not been trained yet.")

        data = torch.from_numpy(data).to(self.device, dtype=torch.float32)
        self.model = self.model.eval()
        total_class_predictions = []

        # predictions (before softmax)
        with torch.no_grad():
            if data.shape[0] < 128:
                predictions = self.model(data)
                class_predict = predictions.argmax(axis=-1)
                class_predict = class_predict.cpu().detach().numpy()
                total_class_predictions += [item for item in class_predict]
            else:
                for i in range(0, data.shape[0], 128):
                    if i+128 < data.shape[0]:
                        cur_data = data[i:i+128, :, :]
                    else:
                        cur_data = data[i:, :, :]
                    predictions = self.model(cur_data)
                    class_predict = predictions.argmax(axis=-1)
                    class_predict = class_predict.cpu().detach().numpy()
                    total_class_predictions += [
                        item for item in class_predict]
        return np.array(total_class_predictions)

    def save(self, path, name='best_model.dic.pkl'):
        if self.model is None:
            raise Exception(
                f"{self.model_name} model has not been trained yet.")
        if not os.path.exists(path):
            os.makedirs(path)
        model_dict = {'state_dict': self.model.state_dict(),
                      'configs': self.trainer_config_para
                      }
        torch.save(model_dict, os.path.join(
            path, name))

    def load(self, path, name='best_model.dic.pkl'):
        self.model = None
        model_dic = torch.load(os.path.join(
            path, name), map_location='cpu')
        self.__dict__.update(model_dic['configs'])  # load trainer_config_para
        self.trainer_config_para = self.__dict__.copy()
        self.model = self.model_f(**self.model_config_para)
        self.model.load_state_dict(model_dic['state_dict'])
        self.model.to(self.device)



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
        super(DGCNNTrainer, self).__init__(num_nodes, num_hiddens, num_classes, batch_size, num_epochs, lr, l1_reg, l2_reg, dropout, device)

    # todo do_epoch
    def train_eeg_part(self, args, group_loader_1, group_loader_2):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_group1 = DGCNN(
                device=device,
                num_nodes=30,
                edge_weight=self.edge_weight,
                edge_idx=self.edge_index,
                num_features=args.num_features,
                num_hiddens=args.num_hiddens,
                num_classes=self.num_classes,
                num_layers=args.num_layers,
            )
        model_group2 = copy.deepcopy(model_group1)
        model_group1.to(device)
        model_group2.to(device)
        self.init_optimizer(model_group1)
        optimizer1 = copy.deepcopy(self.optimizer)
        self.init_optimizer(model_group2)
        optimizer2 = copy.deepcopy(self.optimizer)
        if not args.search:
            print("")
            print("group training")
            for epoch in range(args.num_epochs):
                model_group1.train()
                model_group2.train()
                for batch in group_loader_1:
                    optimizer1.zero_grad()
                    eeg_data = batch['eeg'].to(device, non_blocking=True)
                    label = batch['label'].view(1).long().to(device, non_blocking=True)
                    output = model_group1(eeg_data)
                    loss = F.cross_entropy(output, label)
                    loss.backward()
                    optimizer1.step()
                    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

                for batch in group_loader_2:
                    optimizer2.zero_grad()
                    eeg_data = batch['eeg'].to(device, non_blocking=True)
                    label = batch['label'].view(1).long().to(device, non_blocking=True)
                    output = model_group2(eeg_data)
                    loss = F.cross_entropy(output, label)
                    loss.backward()
                    optimizer2.step()
                    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        return model_group1, model_group2


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