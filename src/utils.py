import random

import torch
import numpy as np
import pytorch_lightning as pl


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    pl.trainer.seed_everything(seed)