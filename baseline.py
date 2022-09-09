import os
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl

class Baseline(pl.LightningModule):
    def __init__(self, pointnet):
        super().__init__()
        self.pointnet = pointnet

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss