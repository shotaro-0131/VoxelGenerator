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
        torch.save(self.model.state_dict(), "test.pth")

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
    val_dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[15554:17498],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)

    # seed=random.sample(range(11000), k=5000)
    dataloader = DataLoader(
            DataSet(data[pdb_id_header].values[:15554],
                    cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
    
    latent_dim = 512

    block_num=4
    f_map = [64, 128, 256, 512]
    drop_out = 0
    kernel_size = 3
    lr = 0.001
    pool_type="ave"
    pool_kernel_size = 2

    hyperparameters = dict(block_num=block_num, kernel_size=kernel_size, f_map=f_map, pool_type=pool_type, pool_kernel_size=pool_kernel_size,
                            latent_dim=latent_dim, in_channel=7, out_channel=4, lr=lr, drop_out=drop_out, gpu_id=gpu_id)

    print(hyperparameters)
    model = UNet(AttributeDict(hyperparameters)).to(device)
    # model = torch.nn.DataParallel(
    #     model) if cfg.training.gpu_num > 1 else model
    trainer = pl.Trainer(max_epochs=cfg.training.epoch,
                            progress_bar_refresh_rate=20,
                            gpus=[gpu_id],
                            logger=False,
                            checkpoint_callback=False,
    )
    model = WrapperModel(model, loss, lr).to(device)
    mlflow.pytorch.autolog(log_models=False)
    with mlflow.start_run(experiment_id=2) as run:
        mlflow.set_tags(hyperparameters)
        trainer.fit(model, dataloader, val_dataloader)

import time

if __name__ == "__main__":

    main()

