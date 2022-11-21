import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import numpy as np

from src.backbones.pointnet2_sem_seg_msg import PointNet2SemMsg


class AffordanceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.fc1 = nn.Linear(300, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, zs):
        """
        zs: [B, n_parts, num_classes, 300]
        """

        B = zs.shape[0]
        n_parts = zs.shape[1]

        preds = torch.zeros(
            (B, n_parts, self.num_classes, 1),
            device='gpu' if torch.cuda.device_count() == 1 else 'cpu')
        for i in range(n_parts):
            for j in range(self.num_classes):
                z = zs[:, i, j, :]
                x = self.drop1(F.relu(self.bn1(self.fc1(z))))
                x = self.drop2(F.relu(self.bn2(self.fc2(x))))
                x = self.fc3(x)
                preds[:, i, j, :] = x

        pred = torch.sigmoid(preds)

        return pred


class PartAttentionModel(nn.Module):
    def __init__(self, num_classes, affordances, num_points):
        super().__init__()
        self.num_classes = num_classes
        embedder = spacy.load('en_core_web_lg')
        affordance_matrix = np.zeros((len(affordances), 300))
        for i, affordance in enumerate(affordances):
            affordance_matrix[i] = embedder(affordance)[0].vector
        self.affordances = torch.tensor(
            affordance_matrix,
            device='cuda'
            if torch.cuda.device_count() == 1 else 'cpu').float()

        self.backbone = PointNet2SemMsg(num_classes=num_classes)
        self.attention = nn.MultiheadAttention(300,
                                               kdim=128,
                                               vdim=128,
                                               num_heads=1,
                                               batch_first=True)

        self.affordance_classifier = AffordanceClassifier(
            num_classes=num_classes)

    def forward(self, part_pcs):
        """
        Dims:
          part_pcs: [B, n_parts, n_points, 3]
        Return dims:
          aff_preds: [B, n_parts, n_affs, 1]
        """
        B = part_pcs.shape[0]
        n_parts = part_pcs.shape[1]
        n_points = part_pcs.shape[2]

        # Extract features for each point in the point cloud
        point_features = torch.zeros(
            (B, n_parts, n_points, 128),
            device='gpu' if torch.cuda.device_count() == 1 else 'cpu')
        for i in range(n_parts):
            part_pc = part_pcs[:, i, :, :]
            if part_pc.count_nonzero() == 0:
                break
            part_features = self.backbone(part_pc.permute(0, 2, 1))
            point_features[:, i, :, :] = part_features

        zs = torch.zeros(
            (B, n_parts, self.num_classes, 300),
            device='gpu' if torch.cuda.device_count() == 1 else 'cpu')
        for i in range(n_parts):
            # Compute attention-weighted embedding for the part with respect to each affordance
            part_features = point_features[:, i, :, :]
            z, _ = self.attention(self.affordances.repeat(B, 1, 1),
                                  part_features, part_features)
            zs[:, i, :, :] = z

        # Use the attention-weighted value vector to predict the part affordances
        aff_preds = self.affordance_classifier(zs)

        # auxiliaries = {'att_weights': att_weights}
        auxiliaries = {}
        return aff_preds, auxiliaries


# TODO:
# Handle part dimensions that are zero
# Make supervision work
# Potential problem: model could just learn to identify parts and predict affordance based on it
#   Potential solution: use the same attention-weighted embedding to segment part and predict its affordance