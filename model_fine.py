from models.u_net import *
import hydra
from omegaconf import DictConfig, omegaconf
from models.loss import VAELoss
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets.voxel_dataset import *
import pandas as pd
import os
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from joblib import parallel_backend

class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    # artifacts = [f.path for f in MlflowClient(
    # ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    # print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


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


@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:

    pl.seed_everything(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    DIR = os.getcwd()
    MODEL_DIR = os.path.join(DIR, 'result')
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    loss = VAELoss()
    dataloader = DataLoader(
        DataSet(data["pdb_id"].values[:8000],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size), batch_size=cfg.training.batch_size, num_workers=8)
    val_dataloader = DataLoader(
        DataSet(data["pdb_id"].values[8000:9000],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size), batch_size=30)

    def objective(trial: optuna.trial.Trial):

        block_num = trial.suggest_int("block_num", 3, 3, log=True)
        kernel_size = trial.suggest_int("kernel_size", 3, 3, log=True)
        pool_type = trial.suggest_categorical("pool", ["max", "ave"])
        pool_kernel_size = trial.suggest_int("pool_kernel_size", 2, 2, log=True)

        f_map = [
            trial.suggest_int("channels_{}".format(i), 32, 1024, log=True) for i in range(block_num)
        ]

        latent_dim = trial.suggest_int("latent_dim", 200, 1028, log=True)

        hyperparameters = dict(block_num=block_num, kernel_size=kernel_size, f_map=f_map, pool_type=pool_type, pool_kernel_size=pool_kernel_size,
                               latent_dim=latent_dim, in_channel=3)
        print(hyperparameters)
        model = UNet(AttributeDict(hyperparameters)).to(device)
        odel = torch.nn.DataParallel(
            model) if cfg.training.gpu_num > 1 else model
        trainer = pl.Trainer(max_epochs=cfg.training.epoch,
                             progress_bar_refresh_rate=20,
                             gpus=[i for i in range(cfg.training.gpu_num)],
                             #plugins='ddp_sharded',
                             accelerator="ddp",
                             callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")])
        model = WrapperModel(model, loss)

        trainer.logger.log_hyperparams(hyperparameters)

        mlflow.pytorch.autolog()
        with mlflow.start_run(experiment_id=2) as run:
            mlflow.set_tags(hyperparameters)
            trainer.fit(model, dataloader, val_dataloader)
        print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        # trainer.fit(model, dataloader, val_dataloader)

        return trainer.callback_metrics["val_loss"].item()

    study = optuna.create_study(direction='minimize')
    #with parallel_backend("multiprocessing", n_jobs=cfg.training.gpu_num):
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
