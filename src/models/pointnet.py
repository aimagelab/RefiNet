from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F


def apply_trans(x, trans):
    """Apply spatial transformations to input

    Attributes:
        x (torch.Tensor): Input Tensor
        trans (torch.nn.Module): Spatial Transformer module

    Returns:
        torch.Tensor: Output Tensor

    """
    x = x.transpose(2, 1)
    x = torch.bmm(x, trans)
    x = x.transpose(2, 1)
    return x


class STN3d(nn.Module):
    """Spatial transformer network in 3D

    Attributes:
        net (torch.nn.Module): Spatial Transformer network implementation
    """
    def __init__(self):
        super(STN3d, self).__init__()
        self.net = STNkd(k=3)

    def forward(self, x):
        return self.net(x)


class STNkd(nn.Module):
    """Spatial transformer network in k-Dimension

    Attributes:
        k (int): Dimensionality

    """
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)      #: torch.nn.Conv1d: 1st 1d conv layer
        self.conv2 = torch.nn.Conv1d(64, 128, 1)    #: torch.nn.Conv1d: 2nd 1d conv layer
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  #: torch.nn.Conv1d: 3rd 1d conv layer
        self.fc1 = nn.Linear(1024, 512)             #: torch.nn.Linear: 1st linear layer
        self.fc2 = nn.Linear(512, 256)              #: torch.nn.Linear: 2nd linear layer
        self.fc3 = nn.Linear(256, k*k)              #: torch.nn.Linear: 3rd linear layer
        self.relu = nn.ReLU()                       #: torch.nn.ReLU: Rectifier activation function

        """Batch Normalization for every layer"""
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batch_size = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, dim=2, keepdim=False)[0]  # "max pool" on points (dim=2) keeping 1024 features

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=torch.float32).view(1, self.k*self.k).repeat(batch_size, 1).to(x.device)
        x = x + iden

        x = x.view(batch_size, self.k, self.k)
        return x

class PointNetFeat(nn.Module):
    """Pointnet feature transform network, used with per-batch stacked pointclouds"""
    def __init__(self, global_feat=True, feature_transform=True):
        super(PointNetFeat, self).__init__()
        self.stn = STNkd(3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x, valid_pts=None):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = apply_trans(x, trans)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = apply_trans(x, trans_feat)
        else:
            trans_feat = None

        point_feat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        z = torch.empty((len(x), 1024), dtype=x.dtype, device=x.device)
        for i in range(len(x)):
            if valid_pts[i, 0] == 0:
                z[i] = torch.max(x[i], dim=1, keepdim=False)[0]
            else:
                z[i] = torch.max(x[i, :, :int(valid_pts[i, 0].item())], dim=1, keepdim=False)[0]
        x = z
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.unsqueeze(-1).repeat(1, 1, n_pts)
            return torch.cat([point_feat, x], 1), trans, trans_feat

class PointPatch(nn.Module):
    """Pointnet full network, used with per-batch stacked pointclouds"""
    def __init__(self, global_feat=True, feature_transform=True, dropout_prob=0.2):
        super(PointPatch, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        self.dropout_prob = dropout_prob

        self.feat = PointNetFeat(global_feat=self.global_feat, feature_transform=self.feature_transform)
        self.fc1 = nn.Linear(1024, 128, bias=False)
        self.fc2 = nn.Linear(128, 3, bias=True)
        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        n_joints = x.size()[1]
        x = x.view(x.size()[0] * n_joints, x.size()[2], x.size()[3]).permute(0, 2, 1)
        x, trans_x, trans_feat_x = self.feat(x[:, :, :-1], x[:, :, -1])
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = x.view(-1, n_joints, 3)

        return x