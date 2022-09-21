from typing import Literal
import os
import random

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.datasets.part_dataset import PartDataset


def get_dataloader(dataset: Literal['train', 'valid', 'test'],
                   small: bool,
                   batch_size: int,
                   pc_size=1024):
    data_path = os.path.join(
        'data', 'PartNet',
        'selected_objects' if not small else 'objects_small', dataset)
    dataset = PartDataset(data_path, pc_size)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pl.trainer.seed_everything(seed)