import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
import torch
import os
from models.ligand_gen import *
INPUT_DIM = 3
OUTPUT_DIM = 3
GRID_SIZE = 20

class ChargeModel(nn.Module):
    def __init__(self, dim):
        super(ChargeModel, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, 4)
        self.batch1 = nn.BatchNorm3d(1)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 16, 4)
        self.conv3 = nn.Conv3d(64, 16, 4)
        self.res1 = ResBlock(32, 6)
        self.res2 = ResBlock(16, 6)
        self.pooling = nn.MaxPool3d(3, stride=2)
        self.relu = nn.ReLU()
        self.enc_var = nn.Linear(dim, dim)
        self.enc_mean = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.batch1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batch2(x)

        x = self.res1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.res2(x)
        x = self.pooling(x)

        x = x.view(batch_size, -1)
        mean = self.relu(self.enc_mean(x))
        var = F.softplus(self.enc_var(x))

        return mean, var


class ConcatModel(nn.Module):
    def __init__(self):
        super(ConcatModel, self).__init__()
        dim = ((GRID_SIZE-8)//2)**3*16
        self.chargeModel = ChargeModel(dim)
        self.gridModel = Encoder(dim)
        self.enc_mean = nn.Linear(dim, dim)
        self.enc_var = nn.Linear(dim, dim)
        self.decoder = Decoder(dim)

    def encoding(self, x):
        batch_size = x.shape[0]
        x1 = x[:, :INPUT_DIM]
        x2 = x[:, INPUT_DIM].view(batch_size, 1, GRID_SIZE, GRID_SIZE, GRID_SIZE)
        mean1, var1 = self.chargeModel(x2)
        mean2, var2 = self.gridModel(x1)
        mean = mean1 + mean2
        var = var1 + var2
        mean = self.enc_mean(mean)
        var = F.softplus(self.enc_var(var))
        return mean, var

    def forward(self, x):
        mean, var = self.encoding(x)
        z = self.sampling(mean, var)
        x = self.decoder(z)
        return x, mean, var

    def get_z(self, x):
        mean, var = self.encoding(x)
        z = self.sampling(mean, var)
        return z

    def sampling(self, mean, var):
        epsilon = torch.randn(mean.shape)
        return mean + torch.sqrt(var) * epsilon


class Trainer:
    def __init__(self, model, dataset, criterion, batch_size=10, epoch=10, display=True, auto_save=True):
        self.model = model
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.dataloader = DataLoader(dataset,  batch_size=batch_size)
        self.batch_size = batch_size
        self.epoch = epoch
        self.criterion = criterion
        self.display = display
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"
        self.auto_save = auto_save
        self.temp_filename = "temp_model"

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
                self.optimizer.zero_grad()
                ligand_pred, mean, var = self.model(protein)

                loss = self.criterion(ligand_pred, ligand, mean, var)
                loss.backward()
                self.optimizer.step()
                average_loss += loss.item()
            self.__log(e+1, average_loss/(i+1))

            if self.auto_save:
                self.save_model(self.temp_filename)

    def save_model(self, filename):
        torch.save(self.model.to('cpu').state_dict(), filename + '.pth')

    def get_model(self):
        return self.model


def get_trainer(root):

    for i, h5filename in enumerate(traverse(root)):
        with h5py.File(h5filename) as h5:
            if i == 0:
                protein_data = np.array(
                    h5['protein'], dtype=np.float32)[:, [0, 1, 2, 6]]
                ligand_data = np.array(
                    h5['ligand'][:, :OUTPUT_DIM], dtype=np.float32)
                # ligand_data = np.array(h5['protein'][:], dtype=np.float32)
                continue
            protein_data = np.concatenate([protein_data, np.array(
                h5['protein'], dtype=np.float32)[:, [0, 1, 2, 6]]])
            ligand_data = np.concatenate([ligand_data, np.array(
                h5['ligand'][:, :OUTPUT_DIM], dtype=np.float32)])

    train_data = DataSet(protein_data, ligand_data)
    model = ConcatModel()
    loss = VAELoss()

    return Trainer(model, train_data, loss)
