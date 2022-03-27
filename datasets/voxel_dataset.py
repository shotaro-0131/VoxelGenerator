from utils.preprocess import random_rotate, random_shift, to_voxel, get_points
import torch
import os
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from torch.utils.data import DataLoader

def get_conf():
    conf = OmegaConf.load(os.path.join(
        hydra.utils.to_absolute_path(""), "params.yaml"))
    return conf

def get_data_dir() -> str:
    conf = get_conf()
    return conf.dataset.data_dir


class DataSet():
    def __init__(self, pdb_id, voxel_size=1, voxel_num=20, is_numpy=False, is_train=False):
        self.pdb_id = pdb_id
        self.is_numpy = is_numpy
        self.voxel_size = voxel_size
        self.voxel_num = voxel_num
        self.data_dir = get_data_dir()
        self.root = hydra.utils.to_absolute_path("")
        self.protein_path = "_pocket.pdb"
        self.ligand_path = "_ligand.sdf"
        self.train = is_train
        self.output_channel = 3

    def __len__(self):
        return len(self.pdb_id)

    def __getitem__(self, index):
        if self.is_numpy:
          input_data, output_data = np.load(os.path.join(*get_conf().dataset.numpy_data_dir, f"v2020-points-{self.pdb_id[index]}.npy"), allow_pickle=True)

          if self.train:
              input_data, output_data = random_rotate([input_data, output_data])
              input_data, output_data = random_shift([input_data, output_data])
          input_data = to_voxel(input_data, self.voxel_num, self.voxel_size)

          output_data = to_voxel(
              output_data, self.voxel_num, self.voxel_size)[:self.output_channel]

          
        else:
          protein_path = os.path.join(
              *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.protein_path)
          ligand_path = os.path.join(
              *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.ligand_path)
          input_data, output_data = get_points(
              protein_path, ligand_path, self.voxel_num, self.voxel_size)
          if self.train:  
              input_data, output_data = random_rotate([input_data, output_data])
              input_data, output_data = random_shift([input_data, output_data])
          input_data = to_voxel(input_data, self.voxel_num, self.voxel_size)

          output_data = to_voxel(
              output_data, self.voxel_num, self.voxel_size)[:self.output_channel]
        input_data = torch.FloatTensor(input_data)
        output_data = torch.FloatTensor(output_data)

        return input_data, output_data

    