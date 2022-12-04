import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.backbones.pointnet2_part_seg_msg import PointNet2PartMsg


class PointNetJointModel(nn.Module):
    def __init__(self, num_slots: int, num_classes: int):
        super().__init__()
        self.num_slots = num_slots
        self.num_classes = num_classes

        self.pointnet = PointNet2PartMsg(num_classes=num_slots)

        self.fc1 = nn.Linear(1024, 512)
        self.ln1 = nn.LayerNorm([num_slots, 512])
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.ln2 = nn.LayerNorm([num_slots, 256])
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]
        cls_label = torch.zeros((B, 16)).to(obj_pc.device)
        cls_label[:, 0] = torch.ones(B).to(obj_pc.device)
        seg_masks, l3_points = self.pointnet(obj_pc, cls_label)

        x = torch.exp(seg_masks) * l3_points.repeat(1, 1, self.num_slots)
        x = x.permute(0, 2, 1)
        x = self.drop1(F.relu(self.ln1(self.fc1(x))))
        x = self.drop2(F.relu(self.ln2(self.fc2(x))))
        x = self.fc3(x)
        aff = torch.sigmoid(x)

        return {
            'affordance': aff,
            'segmentation_mask': seg_masks.permute(0, 2, 1)
        }


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


def divide_segmentation_mask(mask: torch.Tensor):
    B, n_points = mask.shape
    unique = mask.unique()
    n_parts = len(unique)
    divided_mask = torch.zeros((B, n_parts, n_points)).to(mask.device)
    for c in range(n_parts):
        divided_mask[:, c, :] = mask == c

    return divided_mask, n_parts


class PointNetJointLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_seg_mask = pred['segmentation_mask']
        gt_seg_mask = target['segmentation_mask'].long()
        n_slots = pred_seg_mask.shape[1]

        # Fixed order segmentation loss
        # seg_loss = F.nll_loss(pred_seg_mask, gt_seg_mask)

        # Set matching losses (optimal instead of fixed order of slots)
        gt_seg_mask, n_parts = divide_segmentation_mask(gt_seg_mask)
        p = pred_seg_mask.unsqueeze(dim=2).repeat(1, 1, n_parts, 1)
        g = gt_seg_mask.float().unsqueeze(dim=1).repeat(1, n_slots, 1, 1)
        seg_loss_matrix = F.binary_cross_entropy(torch.exp(p),
                                                 g,
                                                 reduction='none')
        seg_loss_matrix = seg_loss_matrix.mean(dim=3)
        seg_loss = min_loss(seg_loss_matrix)

        pred_aff = pred['affordance']
        gt_aff = target['affordance']

        p = pred_aff.unsqueeze(dim=2).repeat(1, 1, n_slots, 1)
        g = gt_aff.float().unsqueeze(dim=1).repeat(1, n_slots, 1, 1)
        aff_loss_matrix = F.binary_cross_entropy(p, g,
                                                 reduction='none').mean(dim=3)
        aff_loss = min_loss(aff_loss_matrix)

        loss = aff_loss + 3 * seg_loss

        return {'seg_loss': seg_loss, 'aff_loss': aff_loss, 'loss': loss}
