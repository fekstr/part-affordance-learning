from typing import Dict, Literal
import itertools
import json

import torch
import numpy as np
from sklearn.decomposition import PCA
from torchmetrics.functional import auroc as pt_auroc, precision_recall_curve
from torchmetrics.functional.classification import multiclass_accuracy, multiclass_precision, multiclass_recall
import matplotlib.pyplot as plt
import umap as umap_lib
import open3d as o3d

from src.evaluation.utils import get_image


def visualize_attention(pc, att_weights):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(pc.cpu().numpy().T)

    att = att_weights.cpu().numpy()
    norm_att = ((att - att.min()) / (att.max() - att.min()))
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
                       [1, 0, 1], [0.5, 1, 0], [1, 0.5, 0], [0.5, 1, 0.5],
                       [1, 1, 0.5]])
    colors = colors[:, None, :]
    for i in range(n_slots):
        mask = masks.argmax(dim=0) == i
        slot_pc = pc[:, mask]
        if affs is not None:
            affordances = (affs[i] >= 0.5).nonzero().flatten()
            affordance_labels = ','.join(
                [index_affordance_map[int(j)] for j in affordances])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(slot_pc.cpu().numpy().T)
        color = colors[i].repeat(slot_pc.shape[1], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)
        if affs is not None:
            pcds.append((pcd, f'slot/{i}/{affordance_labels}'))
        else:
            pcds.append((pcd, f'slot/{i}'))

    ev = o3d.visualization.ExternalVisualizer()
    ev.set(pcds)


def segmentation_accuracy(
    pred: torch.Tensor, target: torch.Tensor, num_classes
) -> Dict[Literal['micro', 'macro', 'worst', 'best'], torch.Tensor]:
    num_classes = int(num_classes)
    trunc_pred = pred[:, :num_classes, :]

    micro = multiclass_accuracy(trunc_pred,
                                target,
                                num_classes=num_classes,
                                average='micro')
    macro = multiclass_accuracy(trunc_pred,
                                target,
                                num_classes=num_classes,
                                average='macro')
    none = multiclass_accuracy(trunc_pred,
                               target,
                               num_classes=num_classes,
                               average='none')
    none_filtered = none.index_select(0, target.unique().int())
    worst = none_filtered.min()
    best = none_filtered.max()

    return {'micro': micro, 'macro': macro, 'worst': worst, 'best': best}


def segmentation_performance_per_class(
    pred: torch.Tensor, target: torch.Tensor, classes: torch.Tensor,
    test_object_name: str
) -> Dict[str, Dict[Literal['accuracy', 'precision', 'recall'], torch.Tensor]]:
    with open('data/PartNet/manual_part_labels.json') as f:
        part_labels = json.load(f)
        part_labels = part_labels[test_object_name]
        part_labels = {v: k for k, v in part_labels.items()}

    accuracy = multiclass_accuracy(pred,
                                   target,
                                   num_classes=pred.shape[1],
                                   average='none')
    precision = multiclass_precision(pred,
                                     target,
                                     num_classes=pred.shape[1],
                                     average='none')
    recall = multiclass_recall(pred,
                               target,
                               num_classes=pred.shape[1],
                               average='none')

    results = {}
    for i in classes.cpu().numpy():
        label = part_labels[i]
        results[label] = {
            'accuracy': accuracy[i],
            'precision': precision[i],
            'recall': recall[i],
        }

    return results


def set_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
