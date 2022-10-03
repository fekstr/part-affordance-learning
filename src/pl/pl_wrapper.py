import os
import io

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import auroc, precision_recall_curve
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


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
    def __init__(self, model, index_affordance_map=None, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.learning_rate = learning_rate
        self.index_affordance_map = index_affordance_map

    def training_step(self, batch, batch_idx):
        obj_pc, part_pc, target, _ = batch
        pred, _ = self.model(obj_pc, part_pc)
        loss = F.cross_entropy(pred, target)
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
        self._pca(features, part_names)
        # Remove dimensions with no positive samples
        # mask = torch.all(targets == 0, dim=0)
        # preds = preds[:, ~mask]
        # targets = targets[:, ~mask]
        # self._compute_auroc(preds, targets)
        # self._precision_recall(preds, targets)
        # self._failures(preds, targets)
        # self._min_max_scores(preds, targets, obj_ids, part_names)

    def _log_image(self, name):
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        self.logger.experiment.log_image(img, name=name)
        plt.clf()

    def _pca(self, features, part_names):
        pca = PCA(2)
        features = features.squeeze(dim=2).cpu()
        components = pca.fit_transform(features)

        pc1 = components[:, 0]
        pc2 = components[:, 1]

        cmap = {}
        i = 0
        for name in part_names:
            if name not in cmap:
                cmap[name] = i
                i += 1
        colors = [cmap[name] for name in part_names]

        plt.scatter(pc1, pc2, c=colors)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        self._log_image('PCA plot')

    def _failures(self, preds, targets):
        """Writes the preds and targets of preds with largest errors"""
        errs = torch.sum(abs(preds - targets), dim=1)
        topk = torch.topk(errs, 5)
        with open('./failures.txt', 'a') as f:
            f.write('Failures\n\n')
            for i in topk.indices:
                round_pred = torch.round(preds[i]).int()
                f.write('Pred: ' + str(round_pred) + '\n')
                f.write('Target: ' + str(targets[i]) + '\n\n')

    def _compute_auroc(self, preds, targets):
        num_classes = targets.shape[1]
        auroc_weighted = auroc(preds, targets, num_classes, average='weighted')
        auroc_macro = auroc(preds, targets, num_classes, average='macro')
        auroc_micro = auroc(preds, targets, num_classes, average='micro')
        self.log('test_auroc_weighted', auroc_weighted)
        self.log('test_auroc_macro', auroc_macro)
        self.log('test_auroc_micro', auroc_micro)

    def _precision_recall(self, preds, targets):
        num_classes = targets.shape[1]
        for i in range(num_classes):
            pred_i = preds[:, i]
            target_i = targets[:, i]
            if all(target_i == 0):
                continue
            name = self.index_affordance_map[i]
            precision, recall, thresholds = precision_recall_curve(
                pred_i, target_i)
            precision = precision.cpu().detach().numpy()
            recall = recall.cpu().detach().numpy()
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve ({name})')
            self._log_image(f'Precision-Recall Curve ({name})')

    def _min_max_scores(self, preds, targets, obj_ids, part_names):
        errs = abs(preds - targets).squeeze(1)
        min_indices = errs.topk(3, largest=False)[1].cpu().numpy()
        max_indices = errs.topk(3, 0)[1].cpu().numpy()
        errs = errs.cpu().numpy()

        min_imgs = [
            get_render(obj_ids[idx], part_names[idx]) for idx in min_indices
        ]
        max_imgs = [
            get_render(obj_ids[idx], part_names[idx]) for idx in max_indices
        ]

        for i, img in enumerate(min_imgs):
            self.logger.experiment.log_image(img,
                                             name='min_error_test_' + str(i))
        for i, img in enumerate(max_imgs):
            self.logger.experiment.log_image(img,
                                             name='max_error_test_' + str(i))

        min_err = round(errs[min_indices[0]], 5)
        max_err = round(errs[max_indices[0]], 5)

        self.log('min_err_test', min_err)
        self.log('max_err_test', max_err)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
