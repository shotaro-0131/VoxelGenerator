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
class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y, z = batch
        p = self(x)
        loss = self.loss(p, z, z)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        p = self.forward(x)
        val_loss = self.loss(p, y, y)
        # self.log("val_loss", val_loss, on_epoch=True)
        return {'val_loss': val_loss}

    def validation_step_end(self, batch_parts):
        # losses from each GPU
        losses = batch_parts['val_loss']
        return torch.mean(losses)

    def validation_epoch_end(self, validation_step_outputs):
        epoch_val_loss = 0
        for out in validation_step_outputs:
            epoch_val_loss += out
        epoch_val_loss = epoch_val_loss/len(validation_step_outputs)
        self.log("epoch_val_loss", epoch_val_loss, on_epoch=True)
        return {'epoch_val_loss': epoch_val_loss}
    def _save_model(self, *_):
        pass 
    def training_epoch_end(self, training_step_outputs):
        torch.save(self.model.state_dict(), "turned_model.pth")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:
    model_type = "multi"
    gpu_id=0
    pl.seed_everything(0)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    print(1.0/27.0)
    loss = VoxelLoss(0)
    pdb_id_header = "pdb_id"
    cell_size=cfg.preprocess.cell_size
    n_cell=cfg.preprocess.grid_size
    # test_used = ["3bkl", "2oi0"]
    # data = data[~data[pdb_id_header].isin(test_used)]
    def get_voxel(target):
        
        ligand_filename=os.path.join(
        hydra.utils.to_absolute_path(""),f"test_data/{target}/crystal_ligand.mol2")
        receptor_filename=os.path.join(
        hydra.utils.to_absolute_path(""),f"test_data/{target}/receptor.pdb")
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
        ligand_voxel=to_voxel(ligand_point, cell_size=cell_size, n_cell=n_cell)
        return protein_voxel

    test_used = pd.read_csv(os.path.join(
                hydra.utils.to_absolute_path(""), cfg.dataset.test_path))
    print(test_used["target"])
    test_data = np.stack([get_voxel(target) for target in test_used["target"].values])
    test_dataloader = DataLoader(
        torch.FloatTensor(test_data), batch_size=1, shuffle=False)

    data = data[~data[pdb_id_header].isin(test_used[pdb_id_header])]
    print(test_data.shape)
            
    val_dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[15554:17498],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)

    # seed=random.sample(range(11000), k=5000)
    dataloader = DataLoader(
            DataSet(data[pdb_id_header].values[:15554],
                    cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
    
    loaded_study = optuna.load_study(study_name="multi", storage="sqlite:///multi.db")
    best_params = loaded_study.best_params
    print(best_params)

    hyperparameters = dict(block_num=best_params["block_num"], kernel_size=best_params["kernel_size"],
    f_map=[best_params[f"channels_{i}"] for i in range(best_params["block_num"])],
    pool_type=best_params["pool"], pool_kernel_size=best_params["pool_kernel_size"],
                            latent_dim=0, in_channel=7, out_channel=3, lr=best_params["lr"], drop_out=0)

    print(hyperparameters)
    model = UNet(AttributeDict(hyperparameters)).to(device)
    # model = torch.nn.DataParallel(
    #     model) if cfg.training.gpu_num > 1 else model
    trainer = pl.Trainer(max_epochs=30,
                            progress_bar_refresh_rate=20,
                            gpus=[i for i in range(cfg.training.gpu_num)],
                            accelerator="ddp",
                            logger=False,
                            # checkpoint_callback=
                            callbacks=[ModelCheckpoint(
                                    dirpath="/gs/hs0/tga-science/murata/models/",
                                    filename=f"best_{model_type}",
                                    save_top_k=1,
                                    verbose=False,
                                    monitor="epoch_val_loss",
                                    mode="min"
                                    ),EarlyStopping(monitor="epoch_val_loss", patience =2)])
    model = WrapperModel(model, loss, best_params["lr"]).to(device)
    mlflow.pytorch.autolog(log_models=False)
    experiment = mlflow.get_experiment_by_name(f"train {model_type}")
    if experiment == None:
        experiment_id = mlflow.create_experiment(f"train {model_type}")
    else:
        experiment_id = experiment.experiment_id
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(hyperparameters)
        trainer.fit(model, dataloader, val_dataloader)

    model = UNet(AttributeDict(hyperparameters))
    model = WrapperModel(model, loss, best_params["lr"])
    model.load_state_dict(torch.load(f"/gs/hs0/tga-science/murata/models/best_{model_type}.ckpt")["state_dict"], strict=False)
    torch.save(model.to("cpu").model.state_dict(), f"/gs/hs0/tga-science/murata/models/best_{model_type}.pth")
    model.load_state_dict(torch.load(f"/gs/hs0/tga-science/murata/models/best_{model_type}.ckpt")["state_dict"], strict=False)
    model.eval()
    model = model.to(device)


    for i, protein in enumerate(test_dataloader):
        input_data = protein.view(1, 7, cfg.preprocess.grid_size, cfg.preprocess.grid_size, cfg.preprocess.grid_size)
        input_data = input_data.to(device)
        output = model(input_data)
        output = output.cpu().detach().numpy()
        np.save(os.path.join(
                hydra.utils.to_absolute_path(""), f"test_data/{test_used['target'].values[i]}/pred_voxel_1.npy"), output[0])

import time

if __name__ == "__main__":

    main()

