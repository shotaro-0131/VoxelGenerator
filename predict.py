from random import betavariate
from models.u_net import *
import hydra
from omegaconf import DictConfig, omegaconf
from models.loss import *
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.voxel_dataset import *
import pandas as pd
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
# from joblib import parallel_backend, Parallel, delayed
from utils.util import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pymol
from utils.preprocess import *
@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:    

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cell_size=cfg.preprocess.cell_size
    n_cell=cfg.preprocess.grid_size

    def get_voxel(target):
            
        ligand_filename=os.path.join(
        hydra.utils.to_absolute_path(""), os.path.join(*cfg.dataset.test.data_dir, target, cfg.dataset.test.ligand_file))
        receptor_filename=os.path.join(
        hydra.utils.to_absolute_path(""), os.path.join(*cfg.dataset.test.data_dir, target, cfg.dataset.test.receptor_file))
        ligand_label="crystal_ligand"
        receptor_label="receptor"
        cmd = pymol.cmd
        cmd.delete("all")
        cmd.load(ligand_filename)
        cmd.load(receptor_filename)
        def get_positions(label):
            mol = []
            cmd.h_add(label)
            cmd.iterate_state(1, label, 'mol.append([x, y, z, elem])', space=locals(), atomic=0)
            return mol
        center=calc_center_ligand(get_positions(ligand_label))
        protein_point = centering(get_positions(receptor_label), center, cell_size*n_cell)
        ligand_point = centering(get_positions(ligand_label), center, cell_size*n_cell)
        protein_voxel = to_voxel(protein_point, cell_size=cell_size, n_cell=n_cell)
        return protein_voxel
        
    test_used = pd.read_csv(os.path.join(
                hydra.utils.to_absolute_path(""), cfg.dataset.test.index_file))
    test_data = np.stack([get_voxel(target) for target in test_used["target"].values])
    test_dataloader = DataLoader(
        torch.FloatTensor(test_data), batch_size=1, shuffle=False)

    try:
        loaded_study = optuna.load_study(study_name=cfg.model.type, storage=f"sqlite:///{cfg.model.type}.db")

    except Exception:
        print(f"Not Found file: {cfg.model.type}.db")
        sys.exit(1)

    best_params = loaded_study.best_params    
    hyperparameters = dict(block_num=best_params["block_num"], kernel_size=best_params["kernel_size"],
    f_map=[best_params[f"channels_{i}"] for i in range(best_params["block_num"])],
    pool_type=best_params["pool"], pool_kernel_size=best_params["pool_kernel_size"],
                            latent_dim=0, in_channel=7, out_channel=cfg.model.output_channel, lr=best_params["lr"], drop_out=0)
    hyperparameters = AttributeDict(hyperparameters)
    model = UNet(hyperparameters)
    model.load_state_dict(torch.load(f"{cfg.model.model_dir}/best_{cfg.model.type}_model.ckpt")["state_dict"], strict=False)
    model.eval()
    model = model.to(device)

    for i, protein in enumerate(test_dataloader):
        input_data = protein.view(1, 7, cfg.preprocess.grid_size, cfg.preprocess.grid_size, cfg.preprocess.grid_size)
        input_data = input_data.to(device)
        output = model(input_data)
        output = output.cpu().detach().numpy()
        np.save(os.path.join(
                hydra.utils.to_absolute_path(""), f"test_data/{test_used['target'].values[i]}/pred_voxel.npy"), output[0])


if __name__ == "__main__":
    main()