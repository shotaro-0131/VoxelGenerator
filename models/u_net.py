import torch.nn as nn
from functools import partial
import torch
from omegaconf import OmegaConf
import os
import hydra
from functools import lru_cache


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=1):
        super(Block, self).__init__()
        self.batch = nn.BatchNorm3d(out_channel)
        self.activ = nn.ReLU()
        conv_in_channels = [in_channel]
        conv_out_channels = [out_channel // 2]
        if conv_out_channels[0] < in_channel:
            conv_out_channels[0] = in_channel
        conv_in_channels.append(conv_out_channels[0])
        conv_out_channels.append(out_channel)
        self.convs = nn.ModuleList(
            [nn.Conv3d(conv_in_channels[i], conv_out_channels[i], kernel_size, stride, padding) for i in range(2)])

    def forward(self, x):
        for i in range(2):
            x = self.convs[i](x)
            x = self.batch(x)
            x = self.activ(x)
        return self.activ(x)


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Upsample, self).__init__()
        self.conv_t = nn.Conv3d(in_channel, out_channel,
                                kernel_size, stride, padding)

    def forward(self, x):
        return self.conv_t(x)


class Encoder(nn.Module):
    def __init__(self, params=None):
        super(Encoder, self).__init__()
        conf = OmegaConf.load(os.path.join(
            hydra.utils.to_absolute_path(""), "params.yaml"))
        self.params = params if params != None else self.get_default_params()
        if self.params.pool_type == "max":
            self.pooling = nn.MaxPool3d(kernel_size=params.pool_kernel_size)
        else:
            self.pooling = nn.AvgPool3d(kernel_size=params.pool_kernel_size)
        self.blocks = [Block(self.params.f_map[i-1] if i != 0 else self.params.in_channel, self.params.f_map[i], self.params.kernel_size)
                       for i in range(self.params.block_num)]
        self.last_dim = (self.calc_voxel_size(conf.preprocess.grid_size,
                                              self.params.kernel_size, self.params.block_num, True))**3*self.params.f_map[-1]
        self.enc_mean = nn.Linear(self.last_dim, params.latent_dim)
        self.enc_var = nn.Linear(self.last_dim, params.latent_dim)
        self.activ = nn.ReLU()

    def calc_voxel_size(self, previous_voxel_size, kernel_size, n, first_layser):
        if n == 0:
            return previous_voxel_size
        if first_layser:
            return self.calc_voxel_size(previous_voxel_size-kernel_size*2, kernel_size, n-1, False)
        else:
            return self.calc_voxel_size(previous_voxel_size//2-kernel_size*2, kernel_size, n-1, False)

    def foward(self, x):
        for i in range(len(self.blocks)):
            if i != 0:
                x = self.pooling(x)
            x = self.blocks[i](x)
        x = x.view(x.size(0), -1)
        mean = self.enc_mean(x)
        var = self.enc_mean(x)
        return self.activ(mean), self.enc_var(var)


class Decoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        conf = OmegaConf.load(os.path.join(
            hydra.utils.to_absolute_path(""), "params.yaml"))
        self.params = params if params != None else self.get_default_params()
        self.upsampling = Upsample()
        self.blocks = [Block(self.params.f_map[i], self.params.f_map[i-1] if i != 0 else self.params.in_channel, self.params.kernel_size)
                       for i in reversed(range(self.params.block_num))]
        self.concat = partial(self._concat, concat=True)
        self.first_dim = (self.calc_voxel_size(conf.preprocess.grid_size,
                                               self.params.kernel_size, self.params.block_num, True))**3*self.params.f_map[-1]
        self.fc = nn.Linear(self.params.latent_dim, self.first_dim)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.params.in_channel, 6, 6, 6)
        for i in range(self.params.block_num):
            if i != 0:
                x = self.upsampling(x)
                x = self.concat(x)
            x = self.blocks[i](x)
        return x

    def calc_voxel_size(self, previous_voxel_size, kernel_size, n, first_layser):
        if n == 0:
            return previous_voxel_size
        if first_layser:
            return self.calc_voxel_size(previous_voxel_size-kernel_size*2, kernel_size, n-1, False)
        else:
            return self.calc_voxel_size(previous_voxel_size//2-kernel_size*2, kernel_size, n-1, False)

    @ staticmethod
    def _concat(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x


class UNet(nn.Module):
    def __init__(self, params):
        self.params = params if params != None else self.get_default_params()
        self.enc = Encoder(params)
        self.dec = Decoder(params)

    def forward(self, x):
        mean, var = self.enc(x)
        z = self.sampling(mean, var)
        x = self.dec(z)
        return x, mean, var

    def sampling(self, mean, var):
        epsilon = torch.randn(mean.shape).to(self.device)
        return mean + torch.sqrt(var) * epsilon
