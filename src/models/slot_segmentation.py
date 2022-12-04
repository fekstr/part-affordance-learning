import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.backbones.pointnet2_sem_seg_msg import PointNet2SemMsg
from src.backbones.pointnet2_part_seg_msg import PointNet2PartMsg
from src.nets.slot_attention import SlotAttention


class Segmenter(nn.Module):
    def __init__(self, num_points, num_affordances):
        super().__init__()
        self.num_points = num_points
        self.num_affordances = num_affordances

        self.fc1 = nn.Linear(128, num_points)

    def forward(self, z):
        x = self.fc1(z)
        x = F.log_softmax(x, dim=1)
        return x


class SlotSegmentationModel(nn.Module):
    def __init__(self, num_classes: int, num_points: int, num_slots: int):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = PointNet2PartMsg(num_classes=num_classes)
        self.slot_attention = SlotAttention(num_slots=num_slots, dim=128)

        self.segmenter = Segmenter(num_points, num_affordances=num_classes)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]

        # Extract features for each point in the point cloud
        cls_label = torch.zeros((B, 16)).to(obj_pc.device)
        cls_label[:, 0] = torch.ones(B).to(obj_pc.device)
        _, point_features = self.backbone(obj_pc, cls_label)
        slots = self.slot_attention(point_features)

        # z = [B, A, 300]

        # Use attention-weighted embedding for the two tasks
        seg_mask = self.segmenter(slots)

        return seg_mask


class SlotSegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_seg_mask = pred
        gt_seg_mask = target['segmentation_mask'].long()

        seg_loss = F.nll_loss(pred_seg_mask, gt_seg_mask)

        loss = seg_loss

        return {'seg_loss': seg_loss, 'loss': loss}