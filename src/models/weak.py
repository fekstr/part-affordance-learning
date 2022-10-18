import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones.pointnet2_features import PointNet2Feature


class WeakModel(nn.Module):
    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = PointNet2Feature(num_classes, False)

        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, part_pcs):
        part_features = []
        for i in range(part_pcs.shape[-1]):
            features = self.backbone(part_pcs[:, :, :, i])
            part_features.append(features)
        agg_feat = torch.cat(part_features, dim=2).mean(dim=2)
        x = agg_feat
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x, features
