import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc


class PLWrapper(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        obj_pc, part_pc, target = batch
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obj_pc, part_pc, target = batch
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        obj_pc, part_pc, targets = batch
        preds, _ = self.model(obj_pc, part_pc)
        return {'preds': preds, 'targets': targets}

    def test_epoch_end(self, batches):
        preds = torch.cat([batch['preds'] for batch in batches], dim=0)
        targets = torch.cat([batch['targets'] for batch in batches], dim=0)
        targets = targets.int()
        # Remove dimensions with no positive samples
        mask = torch.all(targets == 0, dim=0)
        preds = preds[:, ~mask]
        targets = targets[:, ~mask]
        num_classes = targets.shape[1]
        auroc_weighted = auroc(preds, targets, num_classes, average='weighted')
        auroc_macro = auroc(preds, targets, num_classes, average='macro')
        auroc_micro = auroc(preds, targets, num_classes, average='micro')
        self.log('test_auroc_weighted', auroc_weighted)
        self.log('test_auroc_macro', auroc_macro)
        self.log('test_auroc_micro', auroc_micro)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
