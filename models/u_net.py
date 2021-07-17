import torch.nn as nn
from functools import partial
import torch
from omegaconf import OmegaConf
import os
import hydra
from functools import lru_cache


def calc_voxel_size(previous_voxel_size, kernel_size, n, first_layer):
    if n == 0:
        return previous_voxel_size
    if first_layer:
        return calc_voxel_size(previous_voxel_size-(kernel_size-3)*2, kernel_size, n-1, False)
    else:
        return calc_voxel_size(previous_voxel_size//2-(kernel_size-3)*2, kernel_size, n-1, False)

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, encoder, padding=1):
        super(Block, self).__init__()
        self.activ = nn.ReLU()

        if encoder:
            conv_in_channels = [in_channel]
            conv_out_channels = [out_channel // 2]
            if conv_out_channels[0] < in_channel:
                conv_out_channels[0] = in_channel
            conv_in_channels.append(conv_out_channels[0])
            conv_out_channels.append(out_channel)
        else:
            conv_in_channels = [in_channel]
            conv_out_channels = [out_channel // 2]
            conv_in_channels.append(conv_out_channels[0])
            conv_out_channels.append(out_channel)
        self.convs = nn.ModuleList(
            [nn.Conv3d(conv_in_channels[i], conv_out_channels[i], kernel_size, padding=padding) for i in range(2)])
        self.batchs = nn.ModuleList([nn.BatchNorm3d(conv_out_channels[i]) for i in range(2)])

    def forward(self, x):
        for i in range(2):
            x = self.convs[i](x)
            x = self.batchs[i](x)
            x = self.activ(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding=0):
        super(Upsample, self).__init__()
        self.conv_t = nn.ConvTranspose3d(in_channel, out_channel,
                                kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv_t(x)


class Encoder(nn.Module):
    def __init__(self, params=None):
        super(Encoder, self).__init__()
        conf = OmegaConf.load(os.path.join(
            hydra.utils.to_absolute_path(""), "params.yaml"))
        self.params = params if params != None else self.get_default_params()
        if self.params.pool_type == "max":
            self.pooling = nn.MaxPool3d(kernel_size=self.params.pool_kernel_size)
        else:
            self.pooling = nn.AvgPool3d(kernel_size=self.params.pool_kernel_size)
        self.blocks = nn.ModuleList([Block(self.params.f_map[i-1] if i != 0 else self.params.in_channel, self.params.f_map[i], self.params.kernel_size, True)
                       for i in range(self.params.block_num)])
        self.last_dim = (calc_voxel_size(conf.preprocess.grid_size,
                                              self.params.kernel_size, self.params.block_num, True))**3*self.params.f_map[-1]
        self.enc_mean = nn.Linear(self.last_dim, self.params.latent_dim)
        self.enc_var = nn.Linear(self.last_dim, self.params.latent_dim)
        self.activ = nn.ReLU()

    def forward(self, x):
        encoder_features = []
        for i in range(self.params.block_num):
            if i != 0:
                x = self.pooling(x)
            x = self.blocks[i](x)
            encoder_features.append(x)
        x = x.view(x.size(0), -1)
        mean = self.enc_mean(x)
        var = self.enc_var(x)
        return self.activ(mean), torch.sigmoid(var), encoder_features


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        conf = OmegaConf.load(os.path.join(
            hydra.utils.to_absolute_path(""), "params.yaml"))
        self.params = params if params != None else self.get_default_params()
        self.upsamplings = nn.ModuleList([Upsample(self.params.f_map[self.params.block_num-i-1], self.params.f_map[self.params.block_num-i-1], self.params.pool_kernel_size, self.params.pool_kernel_size) for i in range(self.params.block_num)])
        self.blocks = nn.ModuleList([Block(self.params.f_map[i]*2 if i != self.params.block_num-1 else self.params.f_map[i], self.params.f_map[i-1] if i != 0 else self.params.in_channel, self.params.kernel_size, False)
                       for i in reversed(range(self.params.block_num))])
        self.concat = partial(self._concat)
        self.first_voxel = calc_voxel_size(conf.preprocess.grid_size,
                                               self.params.kernel_size, self.params.block_num, True)
        self.first_dim = (self.first_voxel)**3*self.params.f_map[-1]
        self.fc = nn.Linear(self.params.latent_dim, self.first_dim)
        self.active = nn.ReLU()

    def forward(self, x, encoder_features):
        x = self.fc(x)
        x = self.active(x)
        x = x.view(x.size(0), self.params.f_map[-1], self.first_voxel, self.first_voxel, self.first_voxel)
        for i in range(self.params.block_num):
            if i != 0:
                x = self.upsamplings[i](x)
                x = self.active(x)
                x = self.concat(encoder_features[self.params.block_num-i-1], x)
            x = self.blocks[i](x)
        x = torch.sigmoid(x)
        return x


    @ staticmethod
    def _concat(encoder_features, x):
        return torch.cat((encoder_features, x), dim=1)


class UNet(nn.Module):
    def __init__(self, params):
        super(UNet, self).__init__()
        self.params = params if params != None else self.get_default_params()
        self.enc = Encoder(params)
        self.dec = Decoder(params)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, x):
        mean, var, encoder_features = self.enc(x)
        z = self.sampling(mean, var)
        x = self.dec(z, encoder_features)
        return x, mean, var

    def sampling(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon
