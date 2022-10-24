import torch.nn as nn

from src.backbones.pointnet2_cls_msg import PointNet2


class BaselineObjectModel(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = PointNet2(num_classes, False)

    def forward(self, obj_pc):
        pred = self.backbone(obj_pc)
        return pred[0]
