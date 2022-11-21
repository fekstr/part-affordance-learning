import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
import numpy as np

from src.backbones.pointnet2_sem_seg_msg import PointNet2SemMsg


class AffordanceClassifier(nn.Module):
    def __init__(self, num_affordances):
        super().__init__()
        self.num_affordances = num_affordances

        self.fc1 = nn.Linear(300, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, z):
        x = self.fc1(z)
        x = self.bn1(x.permute(0, 2, 1))
        x = self.drop1(F.relu(x.permute(0, 2, 1)))
        x = self.fc2(x)
        x = self.bn2(x.permute(0, 2, 1))
        x = self.drop2(F.relu(x.permute(0, 2, 1)))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class Segmenter(nn.Module):
    def __init__(self, num_points, num_affordances):
        super().__init__()
        self.num_points = num_points
        self.num_affordances = num_affordances

        self.fc1 = nn.Linear(300, num_points)

    def forward(self, z):
        x = self.fc1(z)
        x = torch.sigmoid(x)
        return x


class JointAttentionModel(nn.Module):
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
            num_affordances=num_classes)
        self.segmenter = Segmenter(num_points, num_affordances=num_classes)

    def forward(self, obj_pc):
        B = obj_pc.shape[0]

        # Extract features for each point in the point cloud
        point_features = self.backbone(obj_pc)
        # Compute attention with respect to each affordance
        z, att_weights = self.attention(self.affordances.repeat(B, 1, 1),
                                        point_features, point_features)

        # z = [B, A, 300]

        # Use attention-weighted embedding for the two tasks
        seg_mask = self.segmenter(z)
        aff_preds = self.affordance_classifier(z)

        pred = {'affordance': aff_preds, 'segmentation_mask': seg_mask}
        auxiliaries = {'att_weights': att_weights}
        return pred, auxiliaries


class JointAttentionModelLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_aff = pred['affordance']
        pred_seg_mask = pred['segmentation_mask']
        gt_aff = target['affordance']
        gt_seg_mask = target['segmentation_mask']

        aff_loss = F.binary_cross_entropy(pred_aff, gt_aff.float())

        # Segmentation loss is the BCE for the best fitting part
        n_parts = gt_seg_mask.shape[1]
        n_aff = pred_seg_mask.shape[1]
        n_points = gt_seg_mask.shape[-1]

        # We want to compare the predicted masks to the GT mask for each part
        p = pred_seg_mask.unsqueeze(dim=2).repeat(1, 1, n_parts, 1)
        g = gt_seg_mask.float().unsqueeze(dim=1).repeat(1, n_aff, 1, 1)
        # The GT mask is 0 if the affordance does not exist
        aff_mask = gt_aff == 0
        aff_mask = aff_mask.unsqueeze(dim=3)
        aff_mask = aff_mask.repeat(1, 1, n_parts, n_points)
        g[aff_mask] = 0
        # Parts for which the GT mask is Inf do not exist
        mask = (g.sum(dim=3) == float('Inf'))
        zero_mask = mask.unsqueeze(dim=3).repeat(1, 1, 1, n_points)
        g[zero_mask] = 0
        ce = F.binary_cross_entropy(p, g, reduction='none').mean(dim=3)
        # Do not consider dimensions where GT seg mask is undefined
        ce[mask] = float('Inf')
        ce = ce.amin(2)
        seg_loss = ce.mean()

        loss = 1 * aff_loss + 1 * seg_loss

        return {'aff_loss': aff_loss, 'seg_loss': seg_loss, 'loss': loss}
