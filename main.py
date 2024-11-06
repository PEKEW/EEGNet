import Utils
from models.Utils import get_data_loaders_gender
import models.trainer as Trainer


def main(args):
    print("="*50)
    print("训练正则化图")
    group1, group2 = None, None
    if args.group_mod == 'gender':
        group1, group2 = get_data_loaders_gender(args)
    else:
        # todo  random
        raise ValueError("不支持的分组方法")
    trainer = Trainer.get_trainer(args)
    trainer.train_eeg_part(args, group1, group2)
    print("="*50)
    # return benchmark(args)

if __name__ == '__main__':
    args = Utils.init()
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

