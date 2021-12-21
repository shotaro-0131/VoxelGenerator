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
from joblib import parallel_backend, Parallel, delayed
from utils.util import *

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
        loss = self.loss(p, z, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        p = self.forward(x)
        val_loss = self.loss(p, y, y)
        self.log("val_loss", val_loss, on_epoch=True)
        return {'val_loss': val_loss}
    
    def _save_model(self, *_):
        pass 
    def training_epoch_end(self, training_step_outputs):
        torch.save(self.model.state_dict(), "turned_model.pth")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:

    gpu_id=0
    pl.seed_everything(0)
    device = torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    print(1.0/27.0)
    loss = VoxelLoss(0)
    pdb_id_header = "pdb_id"
    test_used = ["3bkl", "2oi0"]
    data = data[~data[pdb_id_header].isin(test_used)]
    test_used = pd.read_csv(os.path.join(
                hydra.utils.to_absolute_path(""), cfg.dataset.test_path))
    data = data[~data[pdb_id_header].isin(test_used)]
            
    val_dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[15554:17498],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)

    # seed=random.sample(range(11000), k=5000)
    dataloader = DataLoader(
            DataSet(data[pdb_id_header].values[:15554],
                    cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
    
    loaded_study = optuna.load_study(study_name="unet", storage="sqlite:///test2.db")
    best_params = loaded_study.best_params
    print(best_params)

    hyperparameters = dict(block_num=best_params["block_num"], kernel_size=best_params["kernel_size"],
    f_map=[best_params[f"channels_{i}"] for i in range(best_params["block_num"])],
    pool_type=best_params["pool"], pool_kernel_size=best_params["pool_kernel_size"],
                            latent_dim=0, in_channel=7, out_channel=3, lr=best_params["lr"], drop_out=0, gpu_id=gpu_id)

    print(hyperparameters)
    model = UNet(AttributeDict(hyperparameters)).to(device)
    # model = torch.nn.DataParallel(
    #     model) if cfg.training.gpu_num > 1 else model
    trainer = pl.Trainer(max_epochs=best_params["epoch"],
                            progress_bar_refresh_rate=20,
                            gpus=[gpu_id],
                            logger=False,
                            checkpoint_callback=False,
    )
    model = WrapperModel(model, loss, best_params["lr"]).to(device)
    mlflow.pytorch.autolog(log_models=False)
    with mlflow.start_run(experiment_id=2) as run:
        mlflow.set_tags(hyperparameters)
        trainer.fit(model, dataloader, val_dataloader)

import time

if __name__ == "__main__":

    main()

