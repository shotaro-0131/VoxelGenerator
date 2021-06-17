from utils.preprocess import *
import torch
import os
import hydra
import numpy as np

class DataSet:
    def __init__(self, pdb_id, voxel_size=1, voxel_num=20):
        self.pdb_id = pdb_id
        self.voxel_size = voxel_size
        self.voxel_num = voxel_num
        self.data_dir = ["D:", "v2019-other-PL"]
        self.root = hydra.utils.to_absolute_path("")
        self.protein_path = "_pocket.pdb"
        self.ligand_path = "_ligand.mol2"

    def __len__(self):
        return len(self.pdb_id)

    def __getitem__(self, index):
        protein_path = os.path.join(
            self.root, *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.protein_path)
        ligand_path = os.path.join(
            self.root, *self.data_dir, self.pdb_id[index], self.pdb_id[index]+self.ligand_path)

        input_data, output_data = get_points(
            protein_path, ligand_path, self.voxel_num, self.voxel_size)
        input_data = to_voxel(input_data, self.voxel_num, self.voxel_size)[:3]
        output_data = to_voxel(
            output_data, self.voxel_num, self.voxel_size)[:3]
        input_data = np.concatenate([input_data, output_data], axis=0)[1:4]
        output_data = output_data[1:]
        input_data = torch.FloatTensor(input_data)
        output_data = torch.FloatTensor(output_data)
        return input_data, output_data
