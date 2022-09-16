import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from dataset import PartDataset

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

class Baseline(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def training_step(self, batch, batch_idx):
        # x, y = batch
        # x = x.view(x.size(0), -1)
        # z = self.encoder(x)
        # x_hat = self.decoder(z)
        # loss = F.mse_loss(x_hat, x)
        loss = F.nll_loss(pred, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer


if __name__ == '__main__':
    dataset = PartDataset('./data/PartNet/objects_small', 1000)