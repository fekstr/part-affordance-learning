import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import numpy as np

from src.backbones.pointnet2_sem_seg_msg import PointNet2SemMsg


class AttentionModel(nn.Module):
    def __init__(self, num_classes, affordances):
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

        self.fc1 = nn.Linear(300, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]

        # Extract features for each point in the point cloud
        point_features = self.backbone(obj_pc)
        # Compute attention with respect to each affordance
        z, att_weights = self.attention(self.affordances.repeat(B, 1, 1),
                                        point_features, point_features)
        z = z.view(B, 300)
        # Use the attention-weighted value vector to predict the object affordances
        x = self.drop1(F.relu(self.bn1(self.fc1(z))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x, att_weights