import torch.nn as nn

from src.backbones.pointnet2_cls_msg import PointNet2


class BaselineModel(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = PointNet2(num_classes, False)

    def forward(self, obj_pc, part_pc):
        pred = self.backbone(part_pc)
        return pred
