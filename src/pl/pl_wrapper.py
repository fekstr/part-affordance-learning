import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class PLWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        obj_pc = batch['object_point_cloud']
        part_pc = batch['part_point_cloud']
        target = batch['affordances']
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
        self.log('train_loss', loss)
        self.logger.log_metrics({'train/loss': loss}, step=self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        obj_pc = batch['object_point_cloud']
        part_pc = batch['part_point_cloud']
        target = batch['affordances']
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
        self.log('valid_loss', loss)
        self.logger.log_metrics({'valid/loss': loss}, step=self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        obj_pc = batch['object_point_cloud']
        part_pc = batch['part_point_cloud']
        target = batch['affordances']
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
