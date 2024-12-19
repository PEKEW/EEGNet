from torch.utils.data import DataLoader
from Datasets.Datasets import VRSicknessDataset
from Datasets.Samplers import RandomSampler
from typing import Tuple
import torch
from Datasets.DatasetsUtils import SequenceCollator
import numpy as np
import random


def get_data_loaders_eeg_group(args):
    pass


# TODO: improve return single dataloader inside a tuple
def get_data_loaders_eeg_no_group(args) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    group = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH',
    'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']

    def worker_init_fn(worker_id):
        worker_seed = args.rand_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=['eeg'])

    def create_loader(sampler):
        return DataLoader(
            datasets,
            worker_init_fn=worker_init_fn,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=SequenceCollator(
                sequence_length=None, padding_mode='zero', include=['eeg']),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    random_loader_train_1 = create_loader(RandomSampler(
        dataset=datasets, mod='train', group_id=0, group1=group, group2=[], strategy=args.data_sampler_strategy))
    random_loader_test_1 = create_loader(RandomSampler(
        dataset=datasets, mod='test', group_id=0, group1=group, group2=[], strategy=args.data_sampler_strategy))
    random_loader_train_2 = create_loader(RandomSampler(
        dataset=datasets, mod='train', group_id=1, strategy=args.data_sampler_strategy))
    random_loader_test_2 = create_loader(RandomSampler(
        dataset=datasets, mod='test', group_id=1, strategy=args.data_sampler_strategy))

    # group1 => Male group_id = 0
    # group2 => Female group_id = 1
    group1 = ['TYR', 'XSJ', 'CM', 'LMH', 'SHQ', 'LZY', 'TX', 'YHY']
    group2 = ['HZ', 'CYL', 'GKW',  'WJX', 'CWG', 'LZX', 'LJ', 'WZT']
    random_loader_train_1 = create_loader(RandomSampler(
        dataset=datasets, mod='train', group_id=0, group1=group1, group2=group2))
    random_loader_test_1 = create_loader(RandomSampler(
        dataset=datasets, mod='test', group_id=0, group1=group1, group2=group2))
    random_loader_train_2 = create_loader(RandomSampler(
        dataset=datasets, mod='train', group_id=1, group1=group2, group2=group1))
    random_loader_test_2 = create_loader(RandomSampler(
        dataset=datasets, mod='test', group_id=1, group1=group2, group2=group1))
    return random_loader_train_1, random_loader_test_1, random_loader_train_2, random_loader_test_2


def get_data_loader_cnn(args) -> Tuple[DataLoader, DataLoader]:
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
    include = args.mod
    group = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH',
            'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=include)
    train_sampler = RandomSampler(
        dataset=datasets, mod='train', group1=group, group2=[])
    test_sampler = RandomSampler(
        dataset=datasets, mod='test', group1=group, group2=[])
    collate = SequenceCollator(
        sequence_length=None, padding_mode='zero', include=include)
    train_data_loader = create_loader(train_sampler, collate)
    test_data_loader = create_loader(test_sampler, collate)
    return train_data_loader, test_data_loader


# TODO: improve these 3 function can be merged into one
def get_data_loader_all(args) -> Tuple[DataLoader, DataLoader]:
    def create_loader(sampler, collate):
        return DataLoader(
            datasets,
            batch_size=args.mcdis_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    include = ['eeg', 'optical', 'original', 'motion']
    group = ['TYR', 'XSJ', 'CM', 'TX', 'HZ', 'CYL', 'GKW', 'LMH',
            'WJX', 'CWG', 'SHQ', 'YHY', 'LZX', 'LJ', 'WZT', 'LZY']
    datasets = VRSicknessDataset(root_dir=args.root_dir, mod=include)
    train_sampler = RandomSampler(
        dataset=datasets,
        mod='train',
        group1=group,
        group2=[],
        strategy=args.data_sampler_strategy
    )
    test_sampler = RandomSampler(
        dataset=datasets,
        mod='test',
        group1=group,
        group2=[],
        strategy=args.data_sampler_strategy
    )
    collate = SequenceCollator(
        sequence_length=None,
        padding_mode='zero',
        include=include
    )
    train_data_loader = create_loader(train_sampler, collate)
    test_data_loader = create_loader(test_sampler, collate)
    return train_data_loader, test_data_loader
