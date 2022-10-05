# Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch

import torch.nn as nn
from src.backbones.pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNet2Feature(nn.Module):
    def __init__(self, num_class, normal_channel=True):
        super(PointNet2Feature, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3,
                                          [256, 512, 1024], True)

    def forward(self, xyz):
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        return l3_points
