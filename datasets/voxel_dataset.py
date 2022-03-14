from utils.preprocess import *
import torch
import os
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


def get_data_dir() -> str:
    # return cfg.data_dir
    conf = OmegaConf.load(os.path.join(
        hydra.utils.to_absolute_path(""), "params.yaml"))
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

          
    def __len__(self):
        return len(self.pdb_id)

    def __getitem__(self, index):
        if self.is_numpy:
          input_data, output_data = np.load(os.path.join(*self.data_dir).replace("v2020_PL_all", f"v2020-points/v2020-points-{self.pdb_id[index]}.npy"), allow_pickle=True)

          if self.train:
              input_data, output_data = random_rotate([input_data, output_data])
              input_data, output_data = random_shift([input_data, output_data])
          input_data = to_voxel(input_data, self.voxel_num, self.voxel_size)

          output_data = to_voxel(
              output_data, self.voxel_num, self.voxel_size)[:3]

          
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
              output_data, self.voxel_num, self.voxel_size)[:3]
        input_data = torch.FloatTensor(input_data)
        output_data = torch.FloatTensor(output_data)

        return input_data, output_data

import pandas as pd
from torch.utils.data import DataLoader
def get_data_loader():
    data = pd.read_csv("v2020_index.csv")
    pdb_id_header = "pdb_id"
    test_used = ["3bkl", "2oi0"]
    data = data[~data[pdb_id_header].isin(test_used)]
    val_dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[15554:17498],
                0.5, 32, True, False), batch_size=10)

    # seed=random.sample(range(11000), k=5000)
    dataloader = DataLoader(
            DataSet(data[pdb_id_header].values[:15554],
                    0.5, 32, True, True), batch_size=10, num_workers=2)
    return dataloader, val_dataloader
    