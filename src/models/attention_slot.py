import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.backbones.pointnet2_sem_seg_msg import PointNet2SemMsg
from src.nets.slot_attention import SlotAttention


class AffordanceClassifier(nn.Module):
    def __init__(self, num_affordances):
        super().__init__()
        self.num_affordances = num_affordances

        self.fc1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_affordances)

    def forward(self, z):
        x = self.fc1(z)
        x = self.bn1(x.permute(0, 2, 1))
        x = self.drop1(F.relu(x.permute(0, 2, 1)))
        x = self.fc2(x)
        x = self.bn2(x.permute(0, 2, 1))
        x = self.drop2(F.relu(x.permute(0, 2, 1)))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class Segmenter(nn.Module):
    def __init__(self, num_points, num_affordances):
        super().__init__()
        self.num_points = num_points
        self.num_affordances = num_affordances

        self.fc1 = nn.Linear(128, num_points)

    def forward(self, z):
        x = self.fc1(z)
        x = F.softmax(x, dim=1)
        return x


class JointSlotAttentionModel(nn.Module):
    def __init__(self, num_classes: int, affordances: list, num_points: int,
                 num_slots: int):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = PointNet2SemMsg(num_classes=num_classes)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=128)

        self.affordance_classifier = AffordanceClassifier(
            num_affordances=num_classes)
        self.segmenter = Segmenter(num_points, num_affordances=num_classes)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]

        # Extract features for each point in the point cloud
        point_features = self.backbone(obj_pc)
        # Compute attention with respect to each affordance
        # z, att_weights = self.attention(self.affordances.repeat(B, 1, 1),
        #                                 point_features, point_features)
        slots = self.slot_attention(point_features)

        # z = [B, A, 300]

        # Use attention-weighted embedding for the two tasks
        seg_mask = self.segmenter(slots)
        aff_preds = self.affordance_classifier(slots)

        pred = {'affordance': aff_preds, 'segmentation_mask': seg_mask}
        auxiliaries = {'att_weights': None}
        return pred, auxiliaries


def min_loss(loss_matrix):
    B = loss_matrix.shape[0]
    indices = list(
        map(linear_sum_assignment,
            loss_matrix.detach().cpu().numpy()))
    indices = np.array(indices)
    indices = np.transpose(indices, (0, 2, 1))

    full_indices = []
    for i, item in enumerate(indices):
        for idx in item:
            full_indices.append((i, idx[0], idx[1]))

    costs = torch.stack([loss_matrix[idx] for idx in full_indices])
    cost = costs.sum() / B
    return cost


class JointSlotAttentionModelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_aff = pred['affordance']
        pred_seg_mask = pred['segmentation_mask']
        gt_aff = target['affordance']
        gt_seg_mask = target['segmentation_mask']

        n_parts = gt_seg_mask.shape[1]
        n_slots = pred_seg_mask.shape[1]

        p = pred_aff.unsqueeze(dim=2).repeat(1, 1, n_slots, 1)
        g = gt_aff.float().unsqueeze(dim=1).repeat(1, n_slots, 1, 1)
        aff_loss_matrix = F.binary_cross_entropy(p, g,
                                                 reduction='none').mean(dim=3)
        aff_loss = min_loss(aff_loss_matrix)

        # Compute cost between each slot and part mask
        p = pred_seg_mask.unsqueeze(dim=2).repeat(1, 1, n_parts, 1)
        g = gt_seg_mask.float().unsqueeze(dim=1).repeat(1, n_slots, 1, 1)
        # [B, n_slots, n_parts] cost matrix
        seg_loss_matrix = F.binary_cross_entropy(p, g,
                                                 reduction='none').mean(dim=3)
        # seg_loss_matrix = seg_loss_matrix.nan_to_num(float('Inf'))
        seg_loss = min_loss(seg_loss_matrix)

        loss = 1 * aff_loss + 3 * seg_loss

        return {'aff_loss': aff_loss, 'seg_loss': seg_loss, 'loss': loss}
