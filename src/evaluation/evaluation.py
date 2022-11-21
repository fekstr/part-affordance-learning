import itertools

import torch
import numpy as np
from sklearn.decomposition import PCA
from torchmetrics.functional import auroc as pt_auroc, precision_recall_curve
import matplotlib.pyplot as plt
import umap as umap_lib
import open3d as o3d

from src.evaluation.utils import get_image


def visualize_attention(pc, att_weights):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy().T)

    att = att_weights.cpu().numpy()
    norm_att = ((att - att.min()) / (att.max() - att.min()))
    # color = torch.nn.functional.pad(norm_att.T, (0, 2), "constant", 1)
    color = np.repeat(1 - norm_att, 3, axis=0)
    pcd.colors = o3d.utility.Vector3dVector(color.T)

    ev = o3d.visualization.ExternalVisualizer()
    ev.set([pcd])


def visualize_masks(pc: torch.Tensor, masks: torch.Tensor, affs: torch.Tensor,
                    index_affordance_map: dict) -> None:
    """Visualizes the segmentation masks and links them to affordances.
    
    Args:
      pc: one point cloud
      masks: (n_slots, n_points) segmentation masks for the slots of the given point cloud
      affs: (n_slots, n_affs) affordance predictions for each of the slots
    """
    n_slots = masks.shape[0]
    pcds = []
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1],
                       [1, 0, 1]])
    colors = colors[:, None, :]
    for i in range(n_slots):
        mask = masks.argmax(dim=0) == i
        slot_pc = pc[:, mask]
        affordances = (affs[i] >= 0.5).nonzero().flatten()
        affordance_labels = ','.join(
            [index_affordance_map[int(j)] for j in affordances])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(slot_pc.cpu().numpy().T)
        color = colors[i].repeat(slot_pc.shape[1], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
        pcds.append((pcd, f'slot/{i}/{affordance_labels}'))

    ev = o3d.visualization.ExternalVisualizer()
    ev.set(pcds)


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Computes the accuracy of predicted affordance sets.

    The prediction is defined as correct if the set of predicted affordances is
    equal to the set of ground truth affordances (without respect to order).

    Args:
      preds: (B, n_slots, n_aff) tensor with predicted affordance scores by slot
      targets: (B, n_aff, n_aff) tensor with one-hot encoded affordance labels
    """
    n_slots = preds.shape[1]

    # Create all permutations of target rows
    row_perms = list(itertools.permutations(range(n_slots)))
    perm_targets = targets[:, row_perms, :]

    preds_rounded = preds.round().unsqueeze(dim=1).repeat(
        1, perm_targets.shape[1], 1, 1)

    # If pred matrix matches some permutation of the target matrix, the prediction is correct
    matches = (perm_targets == preds_rounded).all(dim=-1).all(dim=-1).any(
        dim=-1).float()

    acc = matches.mean()

    return acc


def auroc(preds, targets):
    num_classes = targets.shape[1]
    auroc_weighted = pt_auroc(preds, targets, num_classes, average='weighted')
    auroc_macro = pt_auroc(preds, targets, num_classes, average='macro')
    auroc_micro = pt_auroc(preds, targets, num_classes, average='micro')
    return {
        'test_auroc_weighted': auroc_weighted,
        'test_auroc_macro': auroc_macro,
        'test_auroc_micro': auroc_micro
    }


def check_overfitting(preds, targets):
    # Computes fraction of rounded predictions with an exact match
    # in the set of unique targets
    np_preds = preds.cpu().numpy()
    np_targets = targets.cpu().numpy()

    matches = np.apply_along_axis(lambda x: all(x), 1,
                                  np_preds.round() == np_targets)
    err_preds = np_preds[~matches, :]

    unique_targets = np.unique(np_targets, axis=0)
    err_pred_matching_target = np.apply_along_axis(
        lambda pred: pred in unique_targets, 1, err_preds.round())
    err_pred_target_overlap = sum(
        err_pred_matching_target) / err_preds.shape[0]
    return {'err_pred_target_overlap': err_pred_target_overlap}


def pca(features, part_names):
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
    fig = plt.gcf()
    return {'PCA plot': get_image(fig)}


def umap(features, part_names):
    reducer = umap_lib.UMAP()
    print('Running UMAP...')
    embedding = reducer.fit_transform(features.cpu().squeeze(dim=2))

    cmap = {}
    i = 0
    for name in part_names:
        if name not in cmap:
            cmap[name] = i
            i += 1
    colors = [cmap[name] for name in part_names]

    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    fig = plt.gcf()
    return {'UMAP plot': get_image(fig)}


# def failures(preds, targets):
#     """Writes the preds and targets of preds with largest errors"""
#     errs = torch.sum(abs(preds - targets), dim=1)
#     topk = torch.topk(errs, 5)
#     with open('./failures.txt', 'a') as f:
#         f.write('Failures\n\n')
#         for i in topk.indices:
#             f.write('Pred: ' + str(round_pred) + '\n')
#             f.write('Target: ' + str(targets[i]) + '\n\n')


def precision_recall(preds, targets):
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
        fig = plt.gcf()
        return {f'Precision-Recall Curve ({name})': get_image(fig)}


# def _min_max_scores(self, preds, targets, obj_ids, part_names):
#     errs = abs(preds - targets).squeeze(1)
#     min_indices = errs.topk(3, largest=False)[1].cpu().numpy()
#     max_indices = errs.topk(3, 0)[1].cpu().numpy()
#     errs = errs.cpu().numpy()

#     min_imgs = [
#         get_render(obj_ids[idx], part_names[idx]) for idx in min_indices
#     ]
#     max_imgs = [
#         get_render(obj_ids[idx], part_names[idx]) for idx in max_indices
#     ]

#     for i, img in enumerate(min_imgs):
#         self.logger.experiment.log_image(img, name='min_error_test_' + str(i))
#     for i, img in enumerate(max_imgs):
#         self.logger.experiment.log_image(img, name='max_error_test_' + str(i))

#     min_err = round(errs[min_indices[0]], 5)
#     max_err = round(errs[max_indices[0]], 5)

#     self.log('min_err_test', min_err)
#     self.log('max_err_test', max_err)
