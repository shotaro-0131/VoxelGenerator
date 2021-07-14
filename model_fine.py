from models.ligand_gen import AutoEncoder
import hydra
from omegaconf import DictConfig, omegaconf
from models.ligand_gen import *
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.voxel_dataset import *
import pandas as pd
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    # artifacts = [f.path for f in MlflowClient(
    # ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    # print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def train(trainer: pl.Trainer, dataloader: DataLoader, val_dataloader: DataLoader, model: pl.LightningModule, tags: dict, experiment_id: int) -> None:
    mlflow.pytorch.autolog()
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(tags)
        trainer.fit(model, dataloader, val_dataloader)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss, val_data=None):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        p, mean, var = self(x)
        loss = self.loss(p, y, mean, var)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        p, mean, var = self.forward(x)
        val_loss = self.loss(p, y, mean, var)
        self.log("val_loss", val_loss, on_epoch=True)
        return {'val_loss': val_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


@hydra.main(config_name="params")
def main(cfg: DictConfig) -> None:
    
    pl.seed_everything(0)

    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, 'result')
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    loss = VAELoss()
    dataloader = DataLoader(
        DataSet(data["pdb_id"].values[:100],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size), batch_size=cfg.training.batch_size, num_workers=8)
    val_dataloader = DataLoader(
            DataSet(data["pdb_id"].values[100:110],
            cfg.preprocess.cell_size, cfg.preprocess.grid_size), batch_size=30)

    def objective(trial: optuna.trial.Trial):

        n_layers = [
            trial.suggest_int("resblocks{}".format(i), 1, 7, log=True) for i in range(2)
        ]
        output_dims = [
            trial.suggest_int("n_channels{}".format(i), 4, 128, log=True) for i in range(3)
        ]
        latent_dims = trial.suggest_int("latent_dim", 200, 1028, log=True) 
        model = AutoEncoder(cfg.preprocess.grid_size, list(map(lambda x: x-1, n_layers)), output_dims, latent_dims)
        trainer = pl.Trainer(max_epochs=1,
                            progress_bar_refresh_rate=20,
                            gpus=cfg.training.gpu_num,
                            callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")])
        model = WrapperModel(model, loss)
        
        hyperparameters = dict(n_layers=n_layers, latent_dims=latent_dims, output_dims=output_dims)
        trainer.logger.log_hyperparams(hyperparameters)
        
        mlflow.pytorch.autolog()
        with mlflow.start_run(experiment_id=2) as run:
            mlflow.set_tags(hyperparameters)
            trainer.fit(model, dataloader, val_dataloader)
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        # trainer.fit(model, dataloader, val_dataloader)

        return trainer.callback_metrics["val_loss"].item()
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


    
    
if __name__ == "__main__":
    main()
    
    
