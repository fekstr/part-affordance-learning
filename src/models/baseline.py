import torch.nn as nn

from src.backbones.pointnet2_cls_msg import PointNet2


class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PointNet2(16, False)

    def forward(self, obj_pc, part_pc):
        pred = self.backbone(part_pc)
        return pred
