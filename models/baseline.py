import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import PartDataset
from pointnet2.backbone import PointNet2

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)

class Baseline(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def training_step(self, batch, batch_idx):
        obj_pc = batch['object_point_cloud']
        part_pc = batch['part_point_cloud']
        target = batch['affordances']
        pred, _ = self.backbone(part_pc)
        loss = F.cross_entropy(pred, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = 1e-3)
        return optimizer


if __name__ == '__main__':
    dataset = PartDataset('./data/PartNet/objects_small', 1024)
    train_loader = DataLoader(dataset, batch_size=2)

    model = Baseline(PointNet2(dataset.num_class, normal_channel=False))

    trainer = pl.Trainer()
    trainer.fit(model=model, train_dataloaders=train_loader)