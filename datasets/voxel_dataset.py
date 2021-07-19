from utils.preprocess import *
import torch
import os
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="..", config_name="params.yaml")
def get_data_dir():
    # return cfg.data_dir
    conf = OmegaConf.load(os.path.join(
        hydra.utils.to_absolute_path(""), "params.yaml"))
    return conf.dataset.data_dir


class DataSet():
    def __init__(self, pdb_id, voxel_size=1, voxel_num=20):
        self.pdb_id = pdb_id
        self.voxel_size = voxel_size
        self.voxel_num = voxel_num
        # self.data_dir = ["D:", "v2019-other-PL"]
        self.data_dir = get_data_dir()
        # self.data_dir = ["/mnt/d", "v2019-other-PL"]
        self.root = hydra.utils.to_absolute_path("")
        self.protein_path = "_pocket.pdb"
        self.ligand_path = "_ligand.sdf"

    def __len__(self):
        return len(self.pdb_id)

    def __getitem__(self, index):
        protein_path = os.path.join(
            *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.protein_path)
        ligand_path = os.path.join(
            *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.ligand_path)
        input_data, output_data = get_points(
            protein_path, ligand_path, self.voxel_num, self.voxel_size)
        input_data = to_voxel(input_data, self.voxel_num, self.voxel_size)[:3]
        output_data = to_voxel(
            output_data, self.voxel_num, self.voxel_size)[:3]
        # input_data = np.concatenate([input_data, output_data], axis=0)[1:4]
        # output_data = output_data[1:]
        input_data = torch.FloatTensor(input_data)
        output_data = torch.FloatTensor(output_data)
        return input_data, output_data

    def random_rotation(points):
        rotate()


class RPNDataSet:
    def __init__(self, pdb_id, voxel_size=1, voxel_num=20):
        self.pdb_id = pdb_id
        self.voxel_size = voxel_size
        self.voxel_num = voxel_num
        # self.data_dir = ["D:", "v2019-other-PL"]
        self.data_dir = get_data_dir()
        self.root = hydra.utils.to_absolute_path("")
        self.protein_path = "_pocket.pdb"
        self.ligand_path = "_ligand.sdf"

    def __len__(self):
        return len(self.pdb_id)

    def __getitem__(self, index):
        protein_path = os.path.join(
            self.root, *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.protein_path)
        ligand_path = os.path.join(
            self.root, *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.ligand_path)

        input_points, output_points = get_points(
            protein_path, ligand_path, self.voxel_num, self.voxel_size)
        input_data = to_voxel(
            input_points, self.voxel_num, self.voxel_size)[:3]

        output_data = to_voxel(
            output_points, int(self.voxel_num/2), self.voxel_size*2)[:3+1]

        seq_data = output_data.reshape((4, -1))

        output_data[3] = np.array([1 if all([seq_data[j][i] == 0 for j in range(3)]) else 0 for i in range(
            seq_data.shape[1])]).reshape((int(self.voxel_num/2), int(self.voxel_num/2), int(self.voxel_num/2)))

        # input_reg = to_delta_xyz(
        #     input_points, self.voxel_num, self.voxel_size)[:3]
        output_reg = to_delta_xyz(
            output_points, self.voxel_num, self.voxel_size)[:3]
        output_reg = torch.FloatTensor(output_reg)
        input_data = torch.FloatTensor(input_data)
        output_data = torch.FloatTensor(output_data)
        return input_data, output_reg, output_data, output_reg
