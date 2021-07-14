import torch
import torch.nn as nn
import torch.nn.functional as F
# import h5py
import numpy as np


class RPN(nn.Module):
    def __init__(self, grid_size):
        super(RPN, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv3d(128, 128, 3, 2, 1) if i % 3 == 0 else nn.Conv3d(128, 128, 3, 1, 1) for i in range(9)])
        self.deconvs = nn.ModuleList(
            [nn.ConvTranspose3d(128, 256, 3, 1, 1),
             nn.ConvTranspose3d(128, 256, 2, 2, 0),
             nn.ConvTranspose3d(128, 256, 4, 4, 1)])
        self.cconv = nn.Conv3d(768, 2, 1, 1)
        self.rconv = nn.Conv3d(768, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # rpn
        # block1 256
        for i in range(3):
            x = self.convs[i](x)
        feat1 = self.deconvs[0](x)
        # block2 256
        for i in range(3, 6):
            x = self.convs[i](x)
        feat2 = self.deconvs[1](x)
        # block3 256
        for i in range(6, 9):
            x = self.convs[i](x)
        feat3 = self.deconvs[2](x)
        feat = torch.cat((feat1, feat2, feat3), 1)
        clf = self.sigmoid(self.cconv(feat))
        reg = self.rconv(feat)
        return clf, reg


class VoxelLoss(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1):
        super(VoxelLoss, self).__init__()
        self.smoothl1loss = nn.SmoothL1Loss(size_average=False)
        self.alpha = alpha
        self.beta = beta

    def forward(self, reg, cls, targets_reg, targets):

        rm_pos_0 = reg.permute(0,2,3,4,1) * torch.unsqueeze(targets[:, 0], -1)
        targets_pos_0 = targets_reg[:, 0] * torch.unsqueeze(targets[:, 0], -1)

        cls_pos_loss = -targets[:, 0] * torch.log(cls[:, 0] + 1e-6)
        cls_pos_loss = cls_pos_loss.sum() / (targets[:, 0].sum() + 1e-6)

        cls_neg_loss = -targets[:, -1] * torch.log(1 - cls[:, 0] + 1e-6)
        cls_neg_loss = cls_neg_loss.sum() / (targets[:, -1].sum() + 1e-6)

        reg_loss = self.smoothl1loss(rm_pos_0, targets_pos_0)
        reg_loss = reg_loss / (targets[:, 0].sum() + 1e-6)
        conf_loss = self.alpha * cls_pos_loss + self.beta * cls_neg_loss
        return conf_loss + reg_loss
