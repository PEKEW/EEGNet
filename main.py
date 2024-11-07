import torch
import Utils
import Utils.Config
from models.Utils import get_data_loaders_gender
import models.trainer as Trainer
from models.DGCNN import DGCNN


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
    print("="*50)
    print("训练正则化图")
    group1, group2 = None, None
    if args.group_mod == 'gender':
        group1, group2 = get_data_loaders_gender(args)
    else:
        # todo  random
        raise ValueError("不支持的分组方法")
    group1_trainer = Trainer.get_trainer(args)
    group2_trainer = Trainer.get_trainer(args)
    group1_trainer._set_data_loader(group1)
    group2_trainer._set_data_loader(group2)
    model_group1 = get_model(args, group1_trainer.edge_weight, group1_trainer.edge_index).to(device)
    model_group2 = get_model(args, group2_trainer.edge_weight, group2_trainer.edge_index).to(device)
    group1_trainer._set_model(model_group1)
    group2_trainer._set_model(model_group2)
    group1_trainer.init_optimizer()
    group2_trainer.init_optimizer()
    for i in range(args.num_epochs):
        print(f"Epoch {i}")
        group1_epoch_metrics = group1_trainer._train_with_eeg(args, i)
        group2_epoch_metrics = group2_trainer._train_with_eeg(args, i)
        print(f"Group1: {group1_epoch_metrics}")
        print(f"Group2: {group2_epoch_metrics}")
    print("="*50)
    # return benchmark(args)

if __name__ == '__main__':
    args = Utils.Config.init()
    main(args)
    exit()













    # if args.train_fold == 'all':
    #     fold_list = np.arange(0, args.n_folds)
    # else:
    #     fold_list = [int(args.train_fold)]
    # result = Results(args)
    # buc = []
    # # gm = GPUManager()
    # args.device_index = 0
    # for i in fold_list:
    #     args_new = copy.deepcopy(args)
    #     args_new.fold_list = [i]
    #     buc.append(main(args_new))
    # para_mean_result_dict = {}
    # if args.subjects_type == 'inter':
    #     for tup in buc:
    #         result.acc_fold_list[tup[0]] = tup[1]
    #         result.subjectsScore[tup[2]] = tup[3]
    # elif args.subjects_type == 'intra':
    #     for tup in buc:
    #         result.acc_fold_list[tup[0]] = tup[1]
    #         result.subjects_results[:, tup[2]] = tup[3]
    #         result.label_val[:, tup[2]] = tup[4]
    # for tup in buc:
    #     if len(para_mean_result_dict) == 0:
    #         para_mean_result_dict = tup[-1]
    #     else:
    #         for k, v in tup[-1].items():
    #             para_mean_result_dict[k]['now_best_acc_train'] += v['now_best_acc_train']
    #             para_mean_result_dict[k]['now_best_acc_val'] += v['now_best_acc_val']
    # for k in para_mean_result_dict.keys():
    #     para_mean_result_dict[k]['now_best_acc_train'] /= len(para_mean_result_dict)
    #     para_mean_result_dict[k]['now_best_acc_val'] /= len(para_mean_result_dict)
    # json.dump({
    #     "para_mean_result_dict": para_mean_result_dict
    # }, open(os.path.join(args.model_path, 'para_mean_result_dict.json'), 'w'))

    # Utils.print_res(args, result)
    # Utils.draw_res(args)

