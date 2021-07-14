import torch.nn as nn
import torch.nn.functional as F
# import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
from datasets.voxel_dataset import DataSet
from models.RPN import *

INPUT_DIM = 3
OUTPUT_DIM = 3


class PermEq(nn.Module):
    def __init__(self, input_dim):
        super(PermEq, self).__init__()
        self.J = torch.ones((input_dim, input_dim))
        self.I = torch.eye(input_dim)
        gamma = torch.empty(input_dim).uniform_(-1/input_dim, 1/input_dim)
        self.gamma = nn.Parameter(gamma)
        lamb = torch.empty(input_dim).uniform_(-1 / input_dim, 1 / input_dim)
        self.lamb = nn.Parameter(lamb)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.lamb * torch.matmul(x, self.I) + \
            self.gamma * torch.matmul(x, self.J)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim, block_num=3):
        super(ResBlock, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv3d(dim, dim, 3, stride=1, padding=1) for i in range(block_num)])
        self.relu = nn.ReLU()
        self.block_num = block_num

    def forward(self, x):
        x1 = x
        for i in range(self.block_num):
            x1 = self.convs[i](x1)
            x1 = self.relu(x1)
            if i % 3 == 2:
                x1 = x + x1
                x = x
        return x1


class Encoder(nn.Module):
    def __init__(self, voxel_size, input_dim=INPUT_DIM, resblocks=[3,3], outputdims=[64, 32], latent_dim=1024):
        super(Encoder, self).__init__()
        self.batch1 = nn.BatchNorm3d(input_dim)
        self.conv1 = nn.Conv3d(input_dim, outputdims[0], 4)
        self.batch2 = nn.BatchNorm3d(outputdims[0])
        self.conv2 = nn.Conv3d(outputdims[0], outputdims[1], 4)
        self.conv3 = nn.Conv3d(outputdims[1], 16, 3, stride=1, padding=1)
        self.res1 = ResBlock(outputdims[0], resblocks[0])
        self.res2 = ResBlock(outputdims[1], resblocks[1])
        self.pooling = nn.MaxPool3d(3, stride=2)
        self.relu = nn.ReLU()
        dim = 16*int((voxel_size-6)/2-1)**3
        self.fc = nn.Linear(dim, dim)
        self.enc_var = nn.Linear(dim, latent_dim)
        self.enc_mean = nn.Linear(dim, latent_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        # x = self.batch1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch2(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = x.view(batch_size, -1)
        # return x
        mean = self.relu(self.enc_mean(x))
        var = F.softplus(self.enc_var(x))
        return mean, var


class Decoder(nn.Module):
    def __init__(self, voxel_size, output_dim=OUTPUT_DIM, resblocks=[3,3], outputdims=[64, 32], latent_dim=1024):
        super(Decoder, self).__init__()
        self.conv_t0 = nn.ConvTranspose3d(
            16, outputdims[0], 3, stride=2, output_padding=1)
        self.conv_t1 = nn.ConvTranspose3d(outputdims[0], outputdims[1], 4)
        self.batch = nn.BatchNorm3d(outputdims[1])
        self.conv_t2 = nn.ConvTranspose3d(outputdims[1], output_dim, 4)
        self.relu = nn.ReLU()
        self.res1 = ResBlock(outputdims[0], resblocks[0])
        self.res2 = ResBlock(outputdims[1], resblocks[1])
        dim = 16*int((voxel_size-6)/2-1)**3
        self.voxel_size = int((voxel_size-6)/2-1)
        # self.conv_t3 = nn.Conv3d(32, OUTPUT_DIM, 3, stride=1, padding=1)
        self.fc = nn.Linear(latent_dim, dim)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.relu(self.fc(x))
        x = x.view(batch_size, 16, self.voxel_size,
                   self.voxel_size, self.voxel_size)
        x = self.conv_t0(x)
        x = self.relu(x)
        # x = self.batch(x)
        x = self.res1(x)
        x = self.conv_t1(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.conv_t2(x)
        # x = F.relu(x)
        # x = self.conv_t3(x)
        x = self.sigmoid(x)
        return x


class VoxelRPN(nn.Module):
    def __init__(self, voxel_size):
        super(VoxelRPN, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        self.voxel_size = voxel_size
        self.enc = Encoder(voxel_size).to(self.device)
        self.dec = Decoder(voxel_size).to(self.device)
        self.rpn = RPN(voxel_size).to(self.device)

    def forward(self, x):
        mean, var = self.enc(x)
        z = self.sampling(mean, var)
        x = self.dec(z)
        clf, reg = self.rpn(x)
        return clf, reg, mean, var

    def get_z(self, x):
        mean, var = self.enc(x)
        z = self.sampling(mean, var)
        return z

    def sampling(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon


class AutoEncoder(nn.Module):
    def __init__(self, voxel_size, n_layers=[3, 3], n_channel=[32, 32], latent_dim=1024):
        super(AutoEncoder, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"

        self.enc = Encoder(voxel_size, INPUT_DIM, n_layers, n_channel, latent_dim).to(self.device)
        self.dec = Decoder(voxel_size, OUTPUT_DIM, n_layers, n_channel, latent_dim).to(self.device)

    def forward(self, x):
        mean, var = self.enc(x)
        z = self.sampling(mean, var)
        x = self.dec(z)
        return x, mean, var

    def get_z(self, x):
        mean, var = self.enc(x)
        z = self.sampling(mean, var)
        return z

    def sampling(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon


class ConModel(nn.Module):
    def __init__(self, voxel_size):
        super(ConModel, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.enc1 = Encoder(voxel_size, 2).to(self.device)
        self.enc2 = Encoder(voxel_size, 1).to(self.device)
        self.dec = Decoder(voxel_size).to(self.device)
        self.voxel_size = voxel_size
        dim = 16*int((voxel_size-6)/2-1)**3
        self.enc_var = nn.Linear(dim, dim)
        self.enc_mean = nn.Linear(dim, dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = x[:, :2]
        x2 = x[:, 2].view(
            batch_size, 1, self.voxel_size, self.voxel_size, self.voxel_size)
        x1 = self.enc1(x1)
        x2 = self.enc2(x2)
        var = F.softplus(self.enc_var(x1+x2))
        mean = self.relu(self.enc_mean(x1+x2))
        z = self.sampling(mean, var)
        x = self.dec(z)
        return x, mean, var

    def get_z(self, x):
        mean1, var1 = self.enc1(x)
        mean2, var2 = self.enc2(x)
        z = self.sampling(mean1+mean2, var1+var2)
        return z

    def sampling(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon


class Trainer:
    def __init__(self, model, dataset, criterion, batch_size=10, epoch=10, display=True, auto_save=True):
        self.model = model
        self.optimizer_enc = optim.Adam(
            self.model.enc.parameters(), lr=0.0001, weight_decay=0.0001)
        self.dataloader = DataLoader(dataset,  batch_size=batch_size)
        self.batch_size = batch_size
        self.epoch = epoch
        self.criterion = criterion
        self.display = display
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.auto_save = auto_save
        self.temp_filename = "temp_model"
        self.optimizer_dec = optim.Adam(
            self.model.dec.parameters(), lr=0.0001, weight_decay=0.0001)

    def __log(self, epoch=0, loss=0):
        if self.display:
            if epoch == 0:
                print("epoch" + " "*10 + "loss")
            else:
                print(str(epoch) + " "*10 + str(loss))

    def run(self):
        self.__log()
        for e in range(self.epoch):
            self.model.train()
            average_loss = 0.0
            for i, data in enumerate(self.dataloader):
                protein, ligand = data
                protein = protein.to(self.device)
                ligand = ligand.to(self.device)
                self.optimizer_enc.zero_grad()
                self.optimizer_dec.zero_grad()
                ligand_pred, mean, var = self.model(protein)
                loss = self.criterion(ligand_pred, ligand, mean, var)
                loss.backward()
                self.optimizer_enc.step()
                self.optimizer_dec.step()
                average_loss += loss.item()
            self.__log(e+1, average_loss/(i+1))

            if self.auto_save:
                self.save_model(self.temp_filename)

    def save_model(self, filename):
        torch.save(self.model, filename)

    def get_model(self):
        return self.model


def traverse(root):
    for d, _, files in os.walk(root):
        for f in files:
            yield os.path.join(d, f)


class VAELoss(nn.Module):
    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha
        self.kldiv = nn.KLDivLoss(reduction="sum")

    def forward(self, outputs, targets, mean=1, var=0):
        reconstruction_loss = F.binary_cross_entropy(outputs, targets)

        kld = -0.5 * torch.sum(1 + torch.log(var) - mean**2 - var)
        return (1-self.alpha)*reconstruction_loss + self.alpha*kld
        # return reconstruction_loss


def get_trainer(root):

    for i, h5filename in enumerate(traverse(root)):
        with h5py.File(h5filename) as h5:
            if i == 0:
                protein_data = np.array(
                    h5['protein'][:, :INPUT_DIM], dtype=np.float32)
                ligand_data = np.array(
                    h5['ligand'][:, :OUTPUT_DIM], dtype=np.float32)
                # ligand_data = np.array(h5['protein'][:], dtype=np.float32)
                continue
            protein_data = np.concatenate([protein_data, np.array(
                h5['protein'][:, :INPUT_DIM], dtype=np.float32)])
            ligand_data = np.concatenate([ligand_data, np.array(
                h5['ligand'][:, :OUTPUT_DIM], dtype=np.float32)])

    train_data = DataSet(protein_data, ligand_data)
    model = AutoEncoder()
    loss = VAELoss()

    return Trainer(model, train_data, loss)
