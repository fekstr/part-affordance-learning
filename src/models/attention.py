import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import numpy as np

from src.backbones.pointnet2_sem_seg_msg import PointNet2SemMsg


class AffordanceClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(300, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, z):
        assert z.shape[1:] == [300, 1]

        x = self.drop1(F.relu(self.bn1(self.fc1(z))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        pred = torch.sigmoid(x)

        assert pred.shape[1:] == [1]
        return pred


class Segmenter(nn.Module):
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points

        num_classes = 2

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, z, point_features):
        assert z.shape[1:] == torch.Size([1, 300])
        assert point_features.shape[1:] == torch.Size([self.num_points, 128])

        x = point_features.permute(0, 2, 1)
        x = self.drop1(F.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x = F.softmax(x, dim=1)
        return x


class AttentionModel(nn.Module):
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

        self.affordance_classifier = AffordanceClassifier()
        self.segmenter = Segmenter(num_points)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]

        # Extract features for each point in the point cloud
        point_features = self.backbone(obj_pc)
        # Compute attention with respect to each affordance
        z, att_weights = self.attention(self.affordances.repeat(B, 1, 1),
                                        point_features, point_features)

        seg_mask = self.segmenter(z, point_features)

        aff_preds = torch.zeros(
            B,
            self.num_classes,
            device='cuda' if torch.cuda.device_count() == 1 else 'cpu')
        for c in range(self.num_classes):
            zc = z[:, c, :].view(B, 300)
            # Use the attention-weighted value vector to predict the object affordances
            aff_pred = self.affordance_classifier(zc)
            aff_preds[:, c] = aff_pred[:, 0]

        auxiliaries = {'att_weights': att_weights}
        return aff_preds, seg_mask, auxiliaries
