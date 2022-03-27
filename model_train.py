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
from utils.util import *
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import random
import sys
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
        x, y = batch
        p = self(x)
        loss = self.loss(p, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        p = self.forward(x)
        val_loss = self.loss(p, y)
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:
    model_type = cfg.model.type
    pl.seed_everything(0)
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    loss = Loss()
    pdb_id_header = "pdb_id"
    train_index_file = os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path)

    try:
        data = pd.read_csv(train_index_file)
    except Exception:
        print(f"Not Found test index file: {train_index_file}")
        sys.exit(1)

    test_index_file = os.path.join(
                hydra.utils.to_absolute_path(""), cfg.dataset.test_path)

    try:
        test_used = pd.read_csv(test_index_file)
    except Exception:
        print(f"Not Found test index file: {test_index_file}")

    data = data[~data[pdb_id_header].isin(test_used[pdb_id_header])]
    random.seed(0)
    random_index=list(range(len(data)))
    random.shuffle(random_index)
    val_index=random_index[:len(data)//5]
    train_index=random_index[len(data)//5:]
    
    val_dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[val_index],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)

    dataloader = DataLoader(
            DataSet(data[pdb_id_header].values[train_index],
                    cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
                    
    try:
        loaded_study = optuna.load_study(study_name=model_type, storage=f"sqlite:///{model_type}.db")

    except Exception:
        print(f"Not Found file: {model_type}.db")
        sys.exit(1)

    best_params = loaded_study.best_params    
    hyperparameters = dict(block_num=best_params["block_num"], kernel_size=best_params["kernel_size"],
    f_map=[best_params[f"channels_{i}"] for i in range(best_params["block_num"])],
    pool_type=best_params["pool"], pool_kernel_size=best_params["pool_kernel_size"],
                            latent_dim=0, in_channel=7, out_channel=cfg.model.output_channel, lr=best_params["lr"], drop_out=0)
    hyperparameters = AttributeDict(hyperparameters)
    lr = best_params["lr"]
    model = UNet(hyperparameters).to(device)    
    model = WrapperModel(model, loss, lr)

    trainer = pl.Trainer(max_epochs=cfg.training.epoch,
                            progress_bar_refresh_rate=20,
                            gpus=[i for i in range(cfg.training.gpu_num)],
                            accelerator="ddp",
                            logger=False,
                            callbacks=[ModelCheckpoint(
                                    dirpath=f"{cfg.model.model_dir}",
                                    filename=f"best_{model_type}_model",
                                    save_top_k=1,
                                    verbose=False,
                                    monitor="epoch_val_loss",
                                    mode="min"
                                    ),EarlyStopping(monitor="epoch_val_loss", patience =2)])

    mlflow.pytorch.autolog(log_models=False)
    experiment = mlflow.get_experiment_by_name(f"train {model_type}")
    if experiment == None:
        experiment_id = mlflow.create_experiment(f"train {model_type}")
    else:
        experiment_id = experiment.experiment_id
        
    with mlflow.start_run(experiment_id=experiment_id) as run:
        if hyperparameters != None:
            mlflow.set_tags(hyperparameters.fields())
        trainer.fit(model, dataloader, val_dataloader)



if __name__ == "__main__":
    main()

