from torch.utils.data import DataLoader
from Datasets.Datasets import VRSicknessDataset, GenderSubjectSamplerMale, GenderSubjectSamplerFemale, RandomSampler
from typing import Tuple
import torch
from Datasets.DatasetsUtils import SequenceCollator
import numpy as np
import random


def get_data_loaders_random(args) -> Tuple[DataLoader]:
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
            collate_fn=SequenceCollator(sequence_length=None, padding_mode='zero',include=['eeg']),
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            drop_last=True
        )
    random_loader_train_1 = create_loader(RandomSampler(dataset=datasets,mod='train',group_id=0))
    random_loader_test_1 = create_loader(RandomSampler(dataset=datasets,mod='test',group_id=0))
    random_loader_train_2 = create_loader(RandomSampler(dataset=datasets,mod='train',group_id=1))
    random_loader_test_2 = create_loader(RandomSampler(dataset=datasets,mod='test',group_id=1))
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