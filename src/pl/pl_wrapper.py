import os
import pickle

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from sklearn.metrics import roc_auc_score

from src.evaluation.evaluation import auroc, pca, umap, visualize_attention, accuracy, visualize_masks


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
    def __init__(self,
                 model,
                 loss,
                 hyperparams=None,
                 index_affordance_map=None,
                 use_test_cache=False,
                 dev=False):
        super().__init__()
        self.dev = dev
        self.hyperparams = hyperparams
        self.model = model
        self.loss = loss
        self.index_affordance_map = index_affordance_map
        self.use_test_cache = use_test_cache

    def training_step(self, batch, batch_idx):
        input, target, _ = batch
        pred, _ = self.model(input)
        loss = self.loss(pred, target)
        self.log('train_loss', loss['loss'])
        self.log('train_aff_loss', loss['aff_loss'])
        self.log('train_seg_loss', loss['seg_loss'])
        return loss['loss']

    # def on_validation_start(self):
    #     self.model.seen_objects = []

    def validation_step(self, batch, batch_idx):
        pcs, target, _ = batch
        pred, _ = self.model(pcs)
        acc = accuracy(pred['affordance'], target['affordance'])
        self.log('val_acc', acc)
        return pcs, pred, target

    def validation_epoch_end(self, batches):
        pass
        # pcs = [batch[0] for batch in batches]
        # preds = [batch[1] for batch in batches]
        # targets = [batch[2] for batch in batches]

        # visualize_masks(pcs[0][1], preds[0]['segmentation_mask'][1],
        #                 preds[0]['affordance'][1], self.index_affordance_map)

        # pred_aff = [pred['affordance'] for pred in preds]
        # pred_aff = torch.cat(pred_aff).squeeze()
        # pred_seg = [pred['segmentation_mask'] for pred in preds]
        # pred_seg = torch.cat(pred_seg)

        # target_aff = [target['affordance'] for target in targets]
        # target_aff = torch.cat(target_aff).squeeze()
        # target_seg = [target['segmentation_mask'] for target in targets]
        # target_seg = torch.cat(target_seg)

        # try:
        #     auc_macro = roc_auc_score(target_aff.cpu(),
        #                               pred_aff.cpu(),
        #                               average='macro',
        #                               multi_class='ovr')
        #     auc_micro = roc_auc_score(target_aff.cpu(),
        #                               pred_aff.cpu(),
        #                               average='micro',
        #                               multi_class='ovr')
        #     auc_weighted = roc_auc_score(target_aff.cpu(),
        #                                  pred_aff.cpu(),
        #                                  average='weighted',
        #                                  multi_class='ovr')
        # except ValueError:
        #     if not self.dev:
        #         raise
        #     else:
        #         auc_macro = 0
        #         auc_micro = 0
        #         auc_weighted = 0

        # self.log('valid_auc_macro', auc_macro, on_epoch=True)
        # self.log('valid_auc_micro', auc_micro, on_epoch=True)
        # self.log('valid_auc_weighted', auc_weighted, on_epoch=True)

    def test_step(self, batch, batch_idx):
        if self.use_test_cache:
            with open('./data/cache/test_cache.pkl', 'rb') as f:
                cached_batch = pickle.load(f)
        pcs, targets, metas = batch
        preds, auxiliaries = self.model(pcs)
        return {
            'pcs': pcs,
            'preds': preds,
            'targets': targets,
            'metas': metas,
            'auxiliaries': auxiliaries
        }

    def test_epoch_end(self, batches):
        pcs = torch.cat([batch['pcs'] for batch in batches], dim=0)
        preds = {}
        preds['affordance'] = torch.cat(
            [batch['preds']['affordance'] for batch in batches], dim=0)
        preds['segmentation_mask'] = torch.cat(
            [batch['preds']['segmentation_mask'] for batch in batches], dim=0)
        targets = {}
        targets['affordance'] = torch.cat(
            [batch['targets']['affordance'] for batch in batches], dim=0)
        targets['segmentation_mask'] = torch.cat(
            [batch['targets']['segmentation_mask'] for batch in batches],
            dim=0)
        auxiliaries = {}
        # for key in batches[0]['auxiliaries']:
        #     cat_item = torch.cat(
        #         [batch['auxiliaries'][key] for batch in batches], dim=0)
        #     auxiliaries[key] = cat_item

        # obj_ids_nested = [batch['part_metas']['obj_id'] for batch in batches]
        # obj_ids = [id for ids in obj_ids_nested for id in ids]
        # part_names_nested = [
        #     batch['part_metas']['part_name'] for batch in batches
        # ]
        # part_names = [id for ids in part_names_nested for id in ids]

        targets['affordance'] = targets['affordance'].int()
        # Remove dimensions with no positive samples
        # mask = torch.all(targets == 0, dim=0)
        # preds = preds[:, ~mask]
        # targets = targets[:, ~mask]
        evaluations = [
            # auroc(preds, targets),
            # pca(features, part_names),
            # umap(features, part_names),
        ]
        for pc, seg_mask, affordance in zip(pcs, preds['segmentation_mask'],
                                            preds['affordance']):
            visualize_masks(pc, seg_mask, affordance,
                            self.index_affordance_map)
        # for i in range(len(pcs)):
        # visualize_attention(pcs[i], auxiliaries['att_weights'][i][0:1, :])
        # visualize_attention(pcs[i], preds['segmentation_mask'][i][0:1, :])
        # visualize_attention(pcs[i], auxiliaries['att_weights'][i][1:2, :])
        # visualize_attention(pcs[i], preds['segmentation_mask'][i][1:2, :])
        # for eval in evaluations:
        #     for name, value in eval.items():
        #         if type(value) is PngImageFile:
        #             self.logger.experiment.log_image(value, name=name)
        #         else:
        #             self.log(name, value)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hyperparams.learning_rate)
        return optimizer
