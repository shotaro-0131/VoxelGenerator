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
        reconstruction_loss = F.binary_cross_entropy(outputs.view(outputs.shape[0], -1), targets.view(outputs.shape[0], -1))

        return reconstruction_loss 

class VoxelLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets, around=None):
        reconstruction_loss = (1-self.alpha)*F.binary_cross_entropy(outputs.view(outputs.shape[0], -1), targets.view(outputs.shape[0], -1))
        if around != None:
            reconstruction_loss += self.alpha*F.binary_cross_entropy(outputs.view(outputs.shape[0], -1), around.view(outputs.shape[0], -1))
        return reconstruction_loss