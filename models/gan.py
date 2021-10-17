import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.voxel_dataset import get_data_loader
from models.loss import VoxelLoss
import numpy as np
from models.u_net import UNet
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import pytorch_lightning as pl
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Discriminator(nn.Module):
    def __init__(self, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv3d(4, d, 4, 2, 1)
        self.conv2 = nn.Conv3d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm3d(d * 2)
        self.conv3 = nn.Conv3d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm3d(d * 4)
        self.conv4 = nn.Conv3d(d * 4, d * 8, 4, 1, 1)
        self.conv4_bn = nn.BatchNorm3d(d * 8)
        self.conv5 = nn.Conv3d(d * 8, 1, 4, 1, 1)
        self.ada_pool = nn.AdaptiveAvgPool3d(1)
        self.relu = nn.LeakyReLU(0.2)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label=None):
        # x = torch.cat([input, label], 1)
        x = input
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        x = self.relu(self.conv4_bn(self.conv4(x)))
        x = torch.sigmoid(self.conv5(x))
        x = self.ada_pool(x).view(x.size(0), 1)

        return x

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class GAN(pl.LightningModule):
    def __init__(self):
        super(GAN, self).__init__()
        self.G = UNet()
        self.D = Discriminator()
        self.adversarial_loss = nn.BCELoss()
        self.recon_loss = VoxelLoss()
        self.automatic_optimization = False

    def forward(self, x):
        return self.G(x)

    def training_step(self, batch, batch_nb, optimizer_idx):
        
        optimizer_G, optimizer_D = self.configure_optimizers()

        x, y, _ = batch
        valid = Variable(Tensor(x.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(x.size(0), 1).fill_(0.0), requires_grad=False)

        fake_output = self.G(x)
        d_z = self.D(fake_output.detach())

        real_loss = self.adversarial_loss(self.D(y), valid)
        fake_loss = self.adversarial_loss(d_z, fake)
        d_loss = (real_loss + fake_loss)/2
        optimizer_D.zero_grad()
        self.manual_backward(d_loss)
        optimizer_D.step()
         
        d_z = self.D(fake_output)
        gan_loss = self.adversarial_loss(d_z, valid)
        recon_loss = self.recon_loss(fake_output, y)
        g_loss = gan_loss + recon_loss
        
        optimizer_G.zero_grad()
        self.manual_backward(g_loss)
        optimizer_G.step()
        return {"G_loss": g_loss, "D_loss": d_loss}

    def training_epoch_end(self, train_step_outputs):
        g_loss = np.mean([val["G_loss"].item() for val in train_step_outputs])
        d_loss = np.mean([val["D_loss"].item() for val in train_step_outputs])
        print(f"G_loss:{g_loss}")
        print(f"D_loss:{d_loss}")


    def validation_step(self, batch, batch_nb):
        x, y = batch
        recon_y = self.G(x)
        d = self.D(recon_y)
        val_loss = self.recon_loss(recon_y, y)
        return {'recon_loss': val_loss, "D_value": d}

    def validation_epoch_end(self, val_step_outputs):
        recon_loss = np.mean([val["recon_loss"].item() for val in val_step_outputs])
        d_value = np.mean([torch.mean(val["D_value"]).item() for val in val_step_outputs])
        print(f"recon_loss:{recon_loss}")
        print(f"D_value:{d_value}")
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1.0e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1.0e-5)
        return g_opt, d_opt
   
    def _save_model(self, *_):
        pass


def main():
    trainer = pl.Trainer(max_epochs=20,
                            progress_bar_refresh_rate=20,
                            gpus=[0],
                            logger=False,
                            checkpoint_callback=False,
    )
    train_dataloader, val_dataloader = get_data_loader()
    model = GAN()
    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == "__main__":
    main()
