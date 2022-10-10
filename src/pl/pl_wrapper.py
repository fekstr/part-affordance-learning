import os

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

from src.evaluation.metrics import auroc, pca, umap


def get_render(obj_id, part_name):
    render_path = os.path.join('data', 'PartNet', 'selected_objects', obj_id,
                               'parts_render')
    candidate_files = [
        f for f in os.listdir(render_path) if f.endswith('.txt')
    ]
    for filename in candidate_files:
        with open(os.path.join(render_path, filename), 'r') as f:
            tokens = f.readline()
            if part_name in tokens.split(' '):
                render_img_path = os.path.join(
                    render_path, filename.replace('.txt', '.png'))
                img = Image.open(render_img_path)
                return img


class PLWrapper(pl.LightningModule):
    def __init__(self, model, hyperparams, index_affordance_map=None):
        super().__init__()
        self.hyperparams = hyperparams
        self.model = model
        self.index_affordance_map = index_affordance_map

    def training_step(self, batch, batch_idx):
        obj_pc, part_pc, target, _ = batch
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(
            pred, target, label_smoothing=self.hyperparams.label_smoothing)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        obj_pc, part_pc, target, _ = batch
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        obj_pc, part_pc, targets, part_metas = batch
        preds, features = self.model(obj_pc, part_pc)
        return {
            'preds': preds,
            'targets': targets,
            'part_metas': part_metas,
            'features': features
        }

    def test_epoch_end(self, batches):
        preds = torch.cat([batch['preds'] for batch in batches], dim=0)
        targets = torch.cat([batch['targets'] for batch in batches], dim=0)
        features = torch.cat([batch['features'] for batch in batches], dim=0)
        obj_ids_nested = [batch['part_metas']['obj_id'] for batch in batches]
        obj_ids = [id for ids in obj_ids_nested for id in ids]
        part_names_nested = [
            batch['part_metas']['part_name'] for batch in batches
        ]
        part_names = [id for ids in part_names_nested for id in ids]

        targets = targets.int()
        # Remove dimensions with no positive samples
        # mask = torch.all(targets == 0, dim=0)
        # preds = preds[:, ~mask]
        # targets = targets[:, ~mask]
        evaluations = [
            # auroc(preds, targets),
            # pca(features, part_names),
            umap(features, part_names),
        ]
        for eval in evaluations:
            for name, value in eval.items():
                if type(value) is PngImageFile:
                    self.logger.experiment.log_image(value, name=name)
                else:
                    self.log(name, value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hyperparams.learning_rate)
        return optimizer
