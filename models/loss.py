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

        kld = -0.5 * torch.sum(1 + torch.log(var) - mean**2 - var)
        return (1-self.alpha)*reconstruction_loss + self.alpha*kld
        # return reconstruction_loss
