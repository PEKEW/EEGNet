from torch.utils.data import Sampler
import random
from sklearn.model_selection import train_test_split
from Utils.Config import Args


class RandomSampler(Sampler):
    init_list = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW',
                 'LMH', 'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']

    def __init__(self, dataset, strategy='down', group1=None, group2=None, mod='train', group_id=0):
        """随机对被试采样

        Args:
            dataset (Dataset): 数据集
            strategy (str, optional): 平衡样本的方法 上采样补全 下采样截断 可选 up|down. Defaults to 'down'.
            group1 (list, optional): group1的被试ID 如果为空则随机填充. Defaults to [].
            group2 (list, optional): 同group1. Defaults to [].
            mod (str, optional): 训练还是测试：可选 train | test. Defaults to 'train'.
            group_id (int, optional): 选择当前的分组是 0 | 1. Defaults to 0.
        """
        self.dataset = dataset
        self.indices = []
        self.mod = mod
        if group1 is None or group2 is None:
            random.shuffle(self.init_list)
            group1 = self.init_list[:len(self.init_list) // 2]
            group2 = self.init_list[len(self.init_list) // 2:]
        self.group1 = group1
        self.group2 = group2
        self.strategy = strategy
        self.id = group_id
        _list = self.group1 if self.id == 0 else self.group2
        positive_indices = []
        negative_indices = []
        for i, (sub_id, slice_id) in enumerate(dataset.samples):
            if sub_id in _list:
                label = dataset.labels[sub_id][f"slice_{slice_id}"]
                if label == 1:
                    positive_indices.append(i)
                else:
                    negative_indices.append(i)
        pos_size = len(positive_indices)
        neg_size = len(negative_indices)
        target_size = max(pos_size, neg_size) if self.strategy == 'up' else min(
            pos_size, neg_size)
        if pos_size > target_size:
            positive_indices = random.sample(positive_indices, target_size)
        elif pos_size < target_size:
            positive_indices = random.choices(positive_indices, k=target_size)
        if neg_size > target_size:
            negative_indices = random.sample(negative_indices, target_size)
        elif neg_size < target_size:
            negative_indices = random.choices(negative_indices, k=target_size)
        self.indices = positive_indices + negative_indices
        random.shuffle(self.indices)
        balanced_labels = [dataset.labels[dataset.samples[i][0]]
                           [f"slice_{dataset.samples[i][1]}"] for i in self.indices]

        train_indices, test_indices = train_test_split(
            self.indices,
            random_state=Args.rand_seed,
            test_size=0.3,
            shuffle=True,
            stratify=balanced_labels
        )

        self.train_indices = train_indices
        self.test_indices = test_indices

    def __iter__(self):
        return iter(self.train_indices if self.mod == 'train' else self.test_indices)

    def __len__(self):
        return len(self.train_indices) if self.mod == 'train' else len(self.test_indices)


# todo trans sampler class 2 new sampler file

class GenderSubjectSampler(Sampler):
    male_sub_list = ['TYR', 'XSJ', 'CM', 'SHQ', 'LMH', 'LZX', 'LJ', 'WZT']
    female_sub_list = ['TX', 'HZ', 'GKW', 'LMH', 'WJX', 'CGW', 'YHY', 'LZY']

    def __init__(self, dataset, strategy='down'):
        """
            strategy (str, optional): down | up. Defaults to 'down'.
        """
        self.dataset = dataset
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class GenderSubjectSamplerMale(GenderSubjectSampler):
    def __init__(self, dataset, strategy='down'):
        super().__init__(dataset, strategy=strategy)
        positive_indices = [
            i for i, (sub_id, slice_id) in enumerate(dataset.samples)
            if (sub_id in self.male_sub_list) and (dataset.labels[sub_id][f"slice_{slice_id}"] == 1)
        ]
        negative_indices = [
            i for i, (sub_id, slice_id) in enumerate(dataset.samples)
            if (sub_id in self.male_sub_list) and (dataset.labels[sub_id][f"slice_{slice_id}"] == 0)
        ]
        pos_size = len(positive_indices)
        neg_size = len(negative_indices)
        target_size = max(pos_size, neg_size) if strategy == 'up' else min(
            pos_size, neg_size)
        if pos_size > target_size:
            positive_indices = random.sample(positive_indices, target_size)
        elif pos_size < target_size:
            positive_indices = random.choices(positive_indices, k=target_size)
        if neg_size > target_size:
            negative_indices = random.sample(negative_indices, target_size)
        elif neg_size < target_size:
            negative_indices = random.choices(negative_indices, k=target_size)

        self.indices = positive_indices + negative_indices
        random.shuffle(self.indices)


class GenderSubjectSamplerFemale(GenderSubjectSampler):
    def __init__(self, dataset, strategy='down'):
        super().__init__(dataset, strategy=strategy)
        positive_indices = [
            i for i, (sub_id, slice_id) in enumerate(dataset.samples)
            if (sub_id in self.female_sub_list) and (dataset.labels[sub_id][f"slice_{slice_id}"] == 1)
        ]
        negative_indices = [
            i for i, (sub_id, slice_id) in enumerate(dataset.samples)
            if (sub_id in self.female_sub_list) and (dataset.labels[sub_id][f"slice_{slice_id}"] == 0)
        ]
        pos_size = len(positive_indices)
        neg_size = len(negative_indices)
        target_size = max(pos_size, neg_size) if strategy == 'up' else min(
            pos_size, neg_size)
        if pos_size > target_size:
            positive_indices = random.sample(positive_indices, target_size)
        elif pos_size < target_size:
            positive_indices = random.choices(positive_indices, k=target_size)
        if neg_size > target_size:
            negative_indices = random.sample(negative_indices, target_size)
        elif neg_size < target_size:
            negative_indices = random.choices(negative_indices, k=target_size)

        self.indices = positive_indices + negative_indices
        random.shuffle(self.indices)


class InterSubjectSampler(Sampler):
    """跨被试采样器"""

    def __init__(self, dataset, fold, sub_list, n_per, is_train=True):
        self.dataset = dataset
        self.fold = fold
        self.sub_list = sub_list
        self.n_subs = len(sub_list)
        self.n_per = n_per
        self.is_train = is_train

        val_start = n_per * fold
        val_end = min(n_per * (fold + 1), self.n_subs)
        val_subs = set(self.sub_list[val_start:val_end])

        self.indices = [
            i for i, (sub_id, _) in enumerate(dataset.samples)
            if (sub_id in val_subs) != is_train
        ]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ExtraSubjectSampler(Sampler):
    def __init__(self, dataset, fold, sub_list, n_per, is_train=True):
        pass

    def __iter__(self):
        raise NotImplementedError(
            "ExtraSubjectSampler is not implemented yet.")
