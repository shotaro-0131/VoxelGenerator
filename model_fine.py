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
from multiprocessing import Process
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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

    print("run_id: {}".format(r.info.run_id))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.best_val = 99

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
        self.log("val_loss", val_loss, on_epoch=True)
        if val_loss < self.best_val:
            self.best_val = val_loss
        self.log("best_loss", self.best_val, on_epoch=True)
        return {'val_loss': val_loss, "best_loss": self.best_val}
    
    def _save_model(self, *_):
        pass 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


GPU_ID=0
@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:
    gpu_id=GPU_ID
    pl.seed_everything(0)
    device = torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda")
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    loss = Loss()
    pdb_id_header = "pdb_id"
    dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[:5000],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
    val_dataloader = DataLoader(
        DataSet(data[pdb_id_header].values[5000:7500],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)

    def objective(trial: optuna.trial.Trial):

        block_num = trial.suggest_int("block_num", 3, 4, log=True)
        kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
        pool_type = trial.suggest_categorical("pool", ["max", "ave"])
        pool_kernel_size = trial.suggest_int("pool_kernel_size", 2, 2, log=True)
        output_channel = 1 if cfg.model.type == "normal" else 3

        f_map = [
            trial.suggest_int("channels_{}".format(i), 32, 252, log=True) for i in range(block_num)
        ]

        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

        hyperparameters = dict(block_num=block_num, kernel_size=kernel_size, f_map=f_map, pool_type=pool_type, pool_kernel_size=pool_kernel_size,
                            in_channel=7, out_channel=output_channel, lr=lr, gpu_id=gpu_id)

        model = UNet(AttributeDict(hyperparameters)).to(device)
        trainer = pl.Trainer(max_epochs=30,
                             progress_bar_refresh_rate=100,
                             gpus=[gpu_id],
                             logger=False,
                             checkpoint_callback=False,
                             callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"), EarlyStopping(monitor="val_loss", patience =2)])
        model = WrapperModel(model, loss, lr).to(device)


        mlflow.pytorch.autolog(log_models=False)
        experiment = mlflow.get_experiment_by_name(f"{cfg.model.type}")
        if experiment == None:
            experiment_id = mlflow.create_experiment(f"{cfg.model.type}")
        else:
            experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.set_tags(hyperparameters)
            trainer.fit(model, dataloader, val_dataloader)

        return trainer.callback_metrics["best_loss"].item()

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction='minimize', load_if_exists=True, pruner=pruner, storage=f"sqlite:///{cfg.model.type}.db", study_name=f"{cfg.model.type}")

    study.optimize(objective, timeout=cfg.training.timeout*60*60, gc_after_trial=True)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    study.trials_dataframe().to_csv("trial_result.csv")

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":

    processes = []
    for rank in range(OmegaConf.load(os.path.join(hydra.utils.to_absolute_path(""), "params.yaml")).training.gpu_num):
        p = torch.multiprocessing.Process(target=main)
        GPU_ID = rank
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

