import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones.pointnet2_part_seg_msg import PointNet2PartMsg


class PointNetSegmentationModel(nn.Module):
    def __init__(self, num_slots: int):
        super().__init__()
        self.pointnet = PointNet2PartMsg(num_classes=num_slots)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]
        cls_label = torch.zeros((B, 16)).to(obj_pc.device)
        cls_label[:, 0] = torch.ones(B).to(obj_pc.device)
        seg_masks, _ = self.pointnet(obj_pc, cls_label)
        return {'segmentation_mask': seg_masks.permute(0, 2, 1)}


class PointNetSegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_seg_mask = pred['segmentation_mask']
        gt_seg_mask = target['segmentation_mask'].long()

        seg_loss = F.nll_loss(pred_seg_mask, gt_seg_mask)

        loss = seg_loss

        return {'seg_loss': seg_loss, 'loss': loss}
