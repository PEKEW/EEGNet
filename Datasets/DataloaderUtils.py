from torch.utils.data import DataLoader
from Datasets.Datasets import VRSicknessDataset
from Datasets.Samplers import *
from typing import Tuple
import torch
from Datasets.DatasetsUtils import SequenceCollator



def get_data_loader_cnn(args) -> DataLoader:
    def create_loader(sampler, collate):
        return DataLoader(
            datasets,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    include = ['original']
    group = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH', 'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=include)
    train_sampler = RandomSampler(dataset=datasets,mod='train', group1=group, group2=[])
    test_sampler = RandomSampler(dataset=datasets,mod='test', group1=group, group2=[])
    collate = SequenceCollator(sequence_length=None, padding_mode='zero',include=include)
    train_data_loader = create_loader(train_sampler, collate)
    test_data_loader = create_loader(test_sampler, collate)
    return train_data_loader, test_data_loader

def get_data_loaders_random(args, datasets=None) -> Tuple[DataLoader]:
    group = args.group
    if datasets is None:
        datasets = VRSicknessDataset(root_dir=args.root_dir, mod=['eeg'])
    def create_loader(sampler):
        return DataLoader(
            datasets,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=SequenceCollator(sequence_length=None, padding_mode='zero',include=['eeg']),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    random_loader_train_1 = create_loader(RandomSampler(dataset=datasets,mod='train',group_id=0, group1=group, group2 = [], strategy=args.data_sampler_strategy))
    random_loader_test_1 = create_loader(RandomSampler(dataset=datasets,mod='test',group_id=0, group1=group, group2 = [], strategy = args.data_sampler_strategy))
    random_loader_train_2 = create_loader(RandomSampler(dataset=datasets,mod='train',group_id=1, strategy=args.data_sampler_strategy))
    random_loader_test_2 = create_loader(RandomSampler(dataset=datasets,mod='test',group_id=1, strategy = args.data_sampler_strategy))
    return random_loader_train_1, random_loader_test_1, random_loader_train_2, random_loader_test_2


def get_data_loaders_gender(args) -> Tuple[DataLoader]:
    raise "这个函数现在废弃了"
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=['eeg'])
    def create_loader(sampler):
        return DataLoader(
            datasets,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=SequenceCollator(sequence_length=None, padding_mode='zero',include=['eeg']),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    male_loader = create_loader(GenderSubjectSamplerMale(datasets))
    female_loader = create_loader(GenderSubjectSamplerFemale(datasets))
    return male_loader, female_loader