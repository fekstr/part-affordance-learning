import os
import pickle
import itertools
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from sklearn.metrics import roc_auc_score

from src.evaluation.evaluation import auroc, pca, umap, visualize_attention, set_accuracy, visualize_masks, segmentation_accuracy, segmentation_performance_per_class


def aggregate_batches(batches):
    aggs = []
    for key in batches[0].keys():
        if type(batches[0][key]) is dict:
            agg = {}
            for k in batches[0][key].keys():
                if type(batches[0][key][k]) is torch.Tensor:
                    agg[k] = torch.cat([batch[key][k] for batch in batches],
                                       dim=0)
                elif type(batches[0][key][k]) is list:
                    agg[k] = list(
                        itertools.chain(*[batch[key][k] for batch in batches]))
        elif type(batches[0][key]) is torch.Tensor:
            agg = torch.cat([batch['pcs'] for batch in batches], dim=0)
        else:
            raise ValueError(
                f'Type {type(batches[0][key])} is not implemented')
        aggs.append(agg)

    return tuple(aggs)


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
        input, target, metas = batch

        ## Temp thing
        # mug_mask = np.array(metas['obj_name']) == 'mug'
        # target['segmentation_mask'][mug_mask, :] += 6

        pred = self.model(input)
        loss = self.loss(pred, target)
        self.log('train_loss', loss['loss'])

        if 'aff_loss' in loss:
            self.log('train_aff_loss', loss['aff_loss'])
        if 'seg_loss' in loss:
            self.log('train_seg_loss', loss['seg_loss'])

        if 'segmentation_mask' in pred:
            seg_acc = segmentation_accuracy(
                pred['segmentation_mask'],
                target['segmentation_mask'],
                num_classes=target['segmentation_mask'].max() + 1)
            self.log('train_seg_acc_micro', seg_acc['micro'])
            self.log('train_seg_acc_macro', seg_acc['macro'])
            self.log('train_seg_acc_worst', seg_acc['worst'])
            self.log('train_seg_acc_best', seg_acc['best'])

        if 'affordance' in pred:
            set_aff_acc = set_accuracy(pred['affordance'],
                                       target['affordance'])
            self.log('train_set_aff_acc', set_aff_acc)

        return loss['loss']

    def validation_step(self, batch, batch_idx):
        pcs, target, metas = batch
        pred = self.model(pcs)
        loss = self.loss(pred, target)
        self.log('val_loss', loss['loss'])

        return {'pcs': pcs, 'pred': pred, 'target': target, 'metas': metas}

    def validation_epoch_end(self, batches):
        pcs, pred, target, metas = aggregate_batches(batches)

        if 'segmentation_mask' in pred:
            seg_acc = segmentation_accuracy(
                pred['segmentation_mask'],
                target['segmentation_mask'],
                num_classes=target['segmentation_mask'].max() + 1)
            self.log('val_seg_acc_micro', seg_acc['micro'])
            self.log('val_seg_acc_macro', seg_acc['macro'])
            self.log('val_seg_acc_worst', seg_acc['worst'])
            self.log('val_seg_acc_best', seg_acc['best'])

        if 'affordance' in pred:
            set_aff_acc = set_accuracy(pred['affordance'],
                                       target['affordance'])
            self.log('val_set_aff_acc', set_aff_acc)

    def test_step(self, batch, batch_idx):
        if self.use_test_cache:
            with open('./data/cache/test_cache.pkl', 'rb') as f:
                cached_batch = pickle.load(f)
        pcs, targets, metas = batch
        preds = self.model(pcs)
        return {
            'pcs': pcs,
            'preds': preds,
            'targets': targets,
            'metas': metas,
        }

    def test_epoch_end(self, batches):
        pcs, pred, target, metas = aggregate_batches(batches)

        evaluations = [
            # auroc(preds, targets),
            # pca(features, part_names),
            # umap(features, part_names),
        ]
        # mug_mask = np.array(metas['obj_name']) == 'mug'
        # target['segmentation_mask'][mug_mask, :] += 6
        # target_seg_mask = torch.zeros(pred['segmentation_mask'].shape).to(
        #     pred['segmentation_mask'].device)
        # for c in range(pred['segmentation_mask'].shape[1]):
        #     target_seg_mask[:,
        #                     c, :] = (target['segmentation_mask'] == c).float()

        # for pc, seg_mask, name in zip(pcs, target_seg_mask, metas['obj_name']):
        #     print(name)
        #     visualize_masks(pc, seg_mask, None, self.index_affordance_map)

        # metrics = segmentation_performance_per_class(
        #     pred['segmentation_mask'], target['segmentation_mask'],
        #     target['segmentation_mask'].unique(), metas['obj_name'][0])
        # for part, values in metrics.items():
        #     for k, v in values.items():
        #         self.log(f'test_{part}_{k}', v)

        for pc, seg_mask, name in zip(pcs, pred['segmentation_mask'],
                                      metas['obj_name']):
            print(name)
            visualize_masks(pc, seg_mask, None, self.index_affordance_map)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.hyperparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[1],
                                                         gamma=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
        # return optimizer
