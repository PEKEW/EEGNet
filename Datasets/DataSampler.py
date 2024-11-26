import os, sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from torch.utils.data import Sampler
import random
from sklearn.model_selection import train_test_split
from Utils.Config import Args

class RandomSampler(Sampler):
    init_list = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH', 'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']
    def __init__(self, dataset, strategy = 'down', group1 = [], group2 = [], mod = 'train', group_id = 0):
        rng = random.Random(Args.rand_seed)
        self.dataset = dataset
        self.indices = []
        self.mod = mod
        if group1 == [] and group2 == []:
            random.shuffle(self.init_list)
            group1 = self.init_list[:len(self.init_list) // 2]
            group2 = self.init_list[len(self.init_list) // 2:]
        elif group1 == [] and group2 != []:
            group1 = [x for x in self.init_list if x not in group2]
        elif group1 != [] and group2 == []:
            group2 = [x for x in self.init_list if x not in group1]
        self.group1 = group1
        self.group2 = group2
        self.strategy = strategy
        self.id = group_id
        _list = self.group1 if self.id == 0 else self.group2
        positive_indices = []
        negative_indices = []
        
        sorted_samples = sorted(enumerate(dataset.samples), 
                                key=lambda x: (x[1][0], x[1][1]))
        self.sorted_indices = [x[0] for x in sorted_samples]
        
        positive_indices = []
        negative_indices = []
        for i in self.sorted_indices:
            sub_id, slice_id = dataset.samples[i]
            if sub_id in _list:
                label = dataset.labels[sub_id][f"slice_{slice_id}"]
                if label == 1:
                    positive_indices.append(i)
                else:
                    negative_indices.append(i)
        
        
        pos_size = len(positive_indices)
        neg_size = len(negative_indices)
        target_size = max(pos_size, neg_size) if self.strategy == 'up' else min(pos_size, neg_size)
        
        
        
        if pos_size > target_size:
            positive_indices = rng.sample(positive_indices, target_size)
        elif pos_size < target_size:
            positive_indices = rng.choices(positive_indices, k=target_size)
        if neg_size > target_size:
            negative_indices = rng.sample(negative_indices, target_size)
        elif neg_size < target_size:
            negative_indices = rng.choices(negative_indices, k=target_size)
        self.indices = positive_indices + negative_indices
        rng.shuffle(self.indices)
        balanced_labels = [dataset.labels[dataset.samples[i][0]][f"slice_{dataset.samples[i][1]}"] for i in self.indices]

        train_indices, test_indices = train_test_split(
            self.indices,
            random_state=Args.rand_seed,
            test_size=0.3, # todo 这里可以小一点
            shuffle=True,
            stratify=balanced_labels
        )
        
        self.train_indices = train_indices
        self.test_indices = test_indices
        
        with open('train_indices.json', 'w') as f:
            samples = [(i, self.dataset.samples[i][0], self.dataset.samples[i][1]) for i in self.train_indices]
            
            for idx, sub_id , slice_id in samples:
                f.write(f"{idx}, {sub_id},{slice_id}\n")
                
        with open('test_indices.json', 'w') as f:
            samples = [(i, self.dataset.samples[i][0], self.dataset.samples[i][1]) for i in self.test_indices]
            for idx, sub_id , slice_id in samples:
                f.write(f"{idx}, {sub_id},{slice_id}\n")


    def __iter__(self):
        return iter(self.train_indices if self.mod == 'train' else self.test_indices)
    def __len__(self):
        return len(self.train_indices) if self.mod == 'train' else len(self.test_indices)
    

class GenderSubjectSampler(Sampler):
    male_sub_list = ['TYR', 'XSJ', 'CM', 'SHQ', 'LMH', 'LZX', 'LJ', 'WZT']
    female_sub_list = ['TX', 'HZ', 'GKW', 'LMH', 'WJX', 'CGW', 'YHY', 'LZY']
    def __init__(self, dataset, strategy = 'down'):
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
    def __init__(self, dataset, strategy = 'down'):
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
        target_size = max(pos_size, neg_size) if strategy == 'up' else min(pos_size, neg_size)
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
    def __init__(self, dataset, strategy = 'down'):
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
        target_size = max(pos_size, neg_size) if strategy == 'up' else min(pos_size, neg_size)
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
    pass

class ExtraSubjectSampler(Sampler):
    pass