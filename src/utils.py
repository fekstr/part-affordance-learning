from typing import Literal
import os
import random

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.datasets.part_dataset import PartDataset


def get_dataloaders(Dataset: PartDataset,
                    small: bool,
                    batch_size: int,
                    pc_size=1024):
    dir = 'objects_small' if small else 'selected_objects'
    data_path = os.path.join('data', 'PartNet', dir)
    dataset = Dataset(data_path, pc_size)
    train_sampler, valid_sampler, test_sampler = dataset.create_split()
    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=train_sampler)
    valid_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=valid_sampler)
    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             sampler=test_sampler)
    return train_loader, valid_loader, test_loader


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pl.trainer.seed_everything(seed)