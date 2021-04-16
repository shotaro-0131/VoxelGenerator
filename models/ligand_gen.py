import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
from datasets.voxel_dataset import DataSet

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
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.batch1 = nn.BatchNorm3d(INPUT_DIM)
        self.conv1 = nn.Conv3d(INPUT_DIM, 32, 4)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 16, 4)
        self.conv3 = nn.Conv3d(64, 16, 4)
        self.res1 = ResBlock(32, 6)
        self.res2 = ResBlock(16, 6)
        self.pooling = nn.MaxPool3d(3, stride=2)
        self.relu = nn.ReLU()
        self.enc_var = PermEq(dim)
        self.enc_mean = PermEq(dim)
        # self.fc = nn.Linear(11*11*11*32, dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.batch1(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch2(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.pooling(x)
        x = x.view(batch_size, -1)
        return x
        # mean = self.relu(self.enc_mean(x))
        # var = F.softplus(self.enc_var(x))
        # return mean, var


class Decoder(nn.Module):
    def __init__(self, dim):
        super(Decoder, self).__init__()
        self.conv_t0 = nn.ConvTranspose3d(16, 64, 3, stride=2, padding=1)
        self.conv_t1 = nn.ConvTranspose3d(64, 64, 3, stride=2)
        self.batch = nn.BatchNorm3d(64)
        self.conv_t2 = nn.ConvTranspose3d(64, 64, 4)
        self.relu = nn.ReLU()
        self.res1 = ResBlock(64, 6)
        self.res2 = ResBlock(64, 6)
        self.conv_t3 = nn.Conv3d(64, OUTPUT_DIM, 4)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size = x.shape[0]
        # x = self.relu(self.fc(x))
        x = x.view(batch_size, 16, 6, 6, 6)
        x = self.conv_t0(x)
        # x = self.batch(x)
        x = self.res1(x)
        x = self.conv_t1(x)
        x = F.relu(x)
        # x = self.conv_t2(x)
        # x = F.relu(x)
        x = self.res2(x)
        x = self.conv_t3(x)
        x = torch.sigmoid(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        dim = 16*6**3

        self.enc = Encoder(dim).to(self.device)
        self.dec = Decoder(dim).to(self.device)

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
        epsilon = torch.randn(mean.shape)
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
    def __init__(self, alpha=0.001):
        super().__init__()
        self.alpha = alpha

    def forward(self, outputs, targets, mean=1, var=0):
        reconstruction_loss = F.binary_cross_entropy(outputs, targets)

        # kld = -0.5 * torch.sum(1 + torch.log(var) - mean**2 - var)

        # return (1-self.alpha)*reconstruction_loss + self.alpha*kld
        return reconstruction_loss


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
