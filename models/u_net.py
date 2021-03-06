import torch.nn as nn
from functools import partial
import torch
from omegaconf import OmegaConf
import os
import hydra
from functools import lru_cache

def get_default_params():
    block_num=4
    f_map = [64, 128, 256, 512]
    drop_out = 0
    kernel_size = 3
    lr = 0.001
    pool_type="ave"
    pool_kernel_size = 2
    gpu_id = 0

    hyperparameters = dict(block_num=block_num, kernel_size=kernel_size, f_map=f_map, pool_type=pool_type, pool_kernel_size=pool_kernel_size,
                            in_channel=7, out_channel=3, lr=lr, drop_out=drop_out, gpu_id=gpu_id)

    return AttributeDict(hyperparameters)

def calc_voxel_size(previous_voxel_size, kernel_size, n, first_layer):
    if n == 0:
        return previous_voxel_size
    if first_layer:
        return calc_voxel_size(previous_voxel_size, kernel_size, n-1, False)
    else:
        return calc_voxel_size(previous_voxel_size//2, kernel_size, n-1, False)

class Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, encoder):
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
            if conv_out_channels[0] == 0:
                conv_out_channels[0] = in_channel
            conv_in_channels.append(conv_out_channels[0])
            conv_out_channels.append(out_channel)

        self.convs = nn.ModuleList(
            [nn.Conv3d(conv_in_channels[i], conv_out_channels[i], kernel_size, padding=(kernel_size-1)//2) for i in range(2)])
        self.batchs = nn.ModuleList([nn.BatchNorm3d(conv_out_channels[i]) for i in range(2)])
        #self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        for i in range(2):
            x = self.convs[i](x)
            if i == 0:
                x = self.batchs[i](x)
                x = self.activ(x)
            # x = self.drop_out(x)
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
        self.blocks = nn.ModuleList([Block(self.params.f_map[i-1] if i != 0 else self.params.in_channel,\
             self.params.f_map[i], self.params.kernel_size, True)
                       for i in range(self.params.block_num)])
        self.activ = nn.ReLU()

    def forward(self, x):
        encoder_features = []
        for i in range(self.params.block_num):
            if i != 0:
                x = self.pooling(x)
            x = self.blocks[i](x)
            x = self.activ(x)
            encoder_features.append(x)
        #x = x.view(x.size(0), -1)

        return x, encoder_features


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        conf = OmegaConf.load(os.path.join(
            hydra.utils.to_absolute_path(""), "params.yaml"))
        self.params = params if params != None else self.get_default_params()
        self.upsamplings = nn.ModuleList([Upsample(self.params.f_map[self.params.block_num-i-1], self.params.f_map[self.params.block_num-i-1], self.params.pool_kernel_size, self.params.pool_kernel_size) for i in range(self.params.block_num)])
        self.blocks = nn.ModuleList([Block(self.params.f_map[i]*2 if i != self.params.block_num-1 else self.params.f_map[i], \
            self.params.f_map[i-1] if i != 0 else self.params.out_channel, self.params.kernel_size, False)
                       for i in reversed(range(self.params.block_num))])
        self.concat = partial(self._concat)
        self.first_voxel = calc_voxel_size(conf.preprocess.grid_size,
                                               self.params.kernel_size, self.params.block_num, True)
        self.first_dim = (self.first_voxel)**3*self.params.f_map[-1]
        #self.drop_out = nn.Dropout(p=params.drop_out)
        self.active = nn.ReLU()

    def forward(self, x, encoder_features):
        for i in range(self.params.block_num):
            if i != 0:
                x = self.upsamplings[i](x)
                x = self.active(x)
                x = self.concat(encoder_features[self.params.block_num-i-1], x)
            x = self.blocks[i](x)
            if i != self.params.block_num-1:
                x = self.active(x)
        x = torch.sigmoid(x)
        return x


    @ staticmethod
    def _concat(encoder_features, x):
        return torch.cat((encoder_features, x), dim=1)

from utils.util import *
class UNet(nn.Module):
    def __init__(self, params=None):
        super(UNet, self).__init__()
        params = params if params != None else get_default_params()
        self.params = params
        self.enc = Encoder(params)
        fields = copy.deepcopy(params.fields())
        dec_params = AttributeDict(fields)
        dec_params.__setstate__([("out_channel", 3 if params.out_channel==1 else 1)])
        self.decs = nn.ModuleList([Decoder(dec_params) for i in range(params.out_channel)])
        if params.gpu_id != None:
            self.device = torch.device("cuda:{}".format(params.gpu_id)) if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enc = self.enc.to(self.device)

    def forward(self, x):
        z, encoder_features = self.enc(x)
        x = [dec(z, encoder_features) for dec in self.decs]
        x = torch.cat(x, dim=1)
        return x

