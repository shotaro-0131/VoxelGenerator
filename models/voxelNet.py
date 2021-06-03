import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
# import pytorch_lightning as pl


class Conv(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super(Conv, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, 32, 3)
        self.batch = nn.BatchNorm3d(32)
        self.maxpool = nn.MaxPool3d(2, 1)
        self.conv2 = nn.Conv3d(32, output_dim, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.dec1 = nn.ConvTranspose3d(input_dim, 32, 3)
        self.batch = nn.BatchNorm3d(32)
        self.dec2 = nn.ConvTranspose3d(32, 10, 3)
        self.dec3 = nn.ConvTranspose3d(10, output_dim, 2)
        self.relu = nn.ReLU()
        self.softMax = nn.Softmax()

    def forward(self, x):
        x = self.dec1(x)
#         x = self.batch(x)
        x = self.relu(x)
        x = self.dec2(x)
        x = self.relu(x)
        x = self.dec3(x)
        x = self.relu(x)
        return x


class VFE(nn.Module):
    def __init__(self, point_num, input_dim, output_dim, device):
        super(VFE, self).__init__()
        self.mlp = nn.Linear(input_dim, int(output_dim/2))
        self.pooling = nn.MaxPool1d(kernel_size=point_num)
        self.output_dim = output_dim
        self.point_num = point_num
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        local_f = self.mlp(x.view(batch_size*self.point_num, -1)).view(batch_size, self.point_num, -1)
#         for i in range(self.point_num):
#             local_f[:, i] = self.mlp(x[:, i])
        global_f = self.pooling(local_f.view(local_f.shape[0], int(
            self.output_dim/2), self.point_num)).squeeze()
        element_f = torch.zeros((x.shape[0], self.point_num, self.output_dim)).to(self.device)
        for i in range(self.point_num):
            element_f[:, i] = torch.cat([local_f[:, i], global_f], dim=1)
        element_f = element_f.view(x.shape[0], self.output_dim, self.point_num)
        return element_f


class VoxelNet(nn.Module):
    def __init__(self, device="cpu"):
        super(VoxelNet, self).__init__()
        self.voxel_num = 20
        self.max_sample_num = 5
        self.output_dim = 3
        self.input_channel = 3
        # must be even number
        self.device = device
        self.point_feature_dim = 200
        self.vfe1 = VFE(self.max_sample_num, 6, self.point_feature_dim, self.device)
        self.vfe2 = VFE(self.max_sample_num, self.point_feature_dim, self.point_feature_dim, self.device)
        self.conv = Conv(self.input_channel, 10)
        self.dec = Decoder(10, self.output_dim)
        self.fc = nn.Linear(self.point_feature_dim, self.input_channel)
        self.point_maxpool = nn.MaxPool1d(self.max_sample_num)

    def forward(self, x):
        # input (BatchSize, voxel_num, point_num, (x, y, z, channel_atom))
        # output (BatchSize, voxel_num, point_num, (x, y, z, channel_atom))
        batch_size = x.shape[0]
        x = self.vfe1(x.view(batch_size*int(pow(self.voxel_num, 3)), -1))
        voxel_feature = self.point_maxpool(self.vfe2(x.view(batch_size*int(pow(self.voxel_num,3)),-1))).squeeze()
#         x = self.fc(voxel_feature).view(batch_size, self.input_channel, self.voxel_num, self.voxel_num, self.voxel_num)
        
#         voxel_feature = self.point_maxpool(self.voxel_wise(inputs.view(batch_size*int(pow(self.voxel_num,3)))).view(batch_size, self.point_feature_dim, self.voxel_num, self.voxel_num, self.voxel_num))
#         for x in range(self.voxel_num):
#             for y in range(self.voxel_num):
#                 for z in range(self.voxel_num):
#                     voxel_feature[:, :, x, y, z] = self.point_maxpool(self.voxel_wise(inputs[:, x, y, z])).squeeze()

        x = self.conv(x)
        x = self.dec(x)
        return x


from utils.preprocess import *
import random
class DataSet(object):
    def __init__(self, protein_file_path, ligand_file_path):
        self.protein_file_path = protein_file_path
        self.ligand_file_path = ligand_file_path
        self.voxel_size = 0.5
        self.voxel_num = 20
        self.data_dir = "../../../v2019-other-PL"
        self.max_sample_num = 5

    def __len__(self):
        return len(self.protein_file_path)

    def __getitem__(self, index):
        protein_path = os.path.join(self.data_dir, self.protein_file_path[index])
        ligand_path = os.path.join(self.data_dir, self.ligand_file_path[index])
        input_data, output_data = get_points(protein_path, ligand_path, self.voxel_num, self.voxel_size)
        output_data = to_voxel(output_data, self.voxel_num, self.voxel_size)[:3]
        input_data = self.sample(input_data)
#         input_data = to_voxel(input_data, self.voxel_num, self.voxel_size)[:3]
        input_data = torch.FloatTensor(input_data)
        output_data = torch.FloatTensor(output_data)
        return input_data, output_data

    def sample(self, data):
        voxel = np.zeros(self.max_sample_num*self.voxel_num *
                         self.voxel_num*self.voxel_num*(3+3))
        voxel = voxel.reshape(self.voxel_num*self.voxel_num *
                              self.voxel_num, self.max_sample_num, 3+3)
        voxel_hash = {}
        data = random.sample(list(data), len(data))
        for d in data:
            if d[3] > 2:
                continue
            vx, vy, vz = self.group(d)
            idx = vx+vy*self.voxel_num+vz*self.voxel_num*self.voxel_num
            if idx in voxel_hash:
                v_idx = voxel_hash[idx]
            else:
                v_idx = 0
            if v_idx == self.max_sample_num-1:
                print("aaaa")
                continue
            # voxel[idx][v_idx] = d
            voxel[idx][v_idx][0] = d[0]
            voxel[idx][v_idx][1] = d[1]
            voxel[idx][v_idx][2] = d[2]
            voxel[idx][v_idx][3] = 1 if d[3]==0 else 0
            voxel[idx][v_idx][4] = 1 if d[3]==1 else 0
            voxel[idx][v_idx][5] = 1 if d[3]==2 else 0
            voxel_hash[idx] = v_idx+1
        return np.array(voxel).reshape(self.voxel_num, self.voxel_num, self.voxel_num, self.max_sample_num, 3+3)

    def group(self, data):
        x = data[0] + self.voxel_num*self.voxel_size/2
        y = data[1] + self.voxel_num*self.voxel_size/2
        z = data[2] + self.voxel_num*self.voxel_size/2
        vx = int(x/self.voxel_size)
        vy = int(y/self.voxel_size)
        vz = int(z/self.voxel_size)
        return vx, vy, vz
