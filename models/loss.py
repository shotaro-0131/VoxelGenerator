import torch.nn as nn
import torch
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.kldiv = nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, mean=1, var=0):
        reconstruction_loss = F.binary_cross_entropy(outputs, targets)

        kld = -0.5 * torch.mean(1 + torch.log(var) - mean**2 - var)
        return reconstruction_loss + kld
        # return reconstruction_loss


class Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        reconstruction_loss = F.binary_cross_entropy(outputs, targets)

        return reconstruction_loss 