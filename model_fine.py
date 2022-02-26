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
    # artifacts = [f.path for f in MlflowClient(
    # ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    # print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = self.model.to(device)
        self.lr = lr
        self.best_val = 99

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
        self.log("val_loss", val_loss, on_epoch=True)
        if val_loss < self.best_val:
            self.best_val = val_loss
        self.log("best_loss", self.best_val, on_epoch=True)
        return {'val_loss': val_loss, "best_loss": self.best_val}
    
    def _save_model(self, *_):
        pass 
    #def training_epoch_end(self, training_step_outputs):
        #torch.save(self.model.state_dict(), "test.pth")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        # discard the version number
        # convert the loss in the proper format
        items["gpu_id"] = GPU_ID
        items["node_id"] = RANK
        return items

GPU_ID=0
RANK=0
@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:
    is_success = False
    while not is_success:
        try:

            gpu_id=GPU_ID
            pl.seed_everything(0)
            device = torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
            # device = torch.device("cuda")
            data = pd.read_csv(os.path.join(
                hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
            loss = VoxelLoss(1.0/27.0)
            # seed=random.sample(range(10000), k=5000)
            pdb_id_header = "pdb_id"
            # test_used = pd.read_csv(os.path.join(
            #     hydra.utils.to_absolute_path(""), cfg.dataset.test_path))
            # data = data[~data[pdb_id_header].isin(test_used)]
            print(data)
            dataloader = DataLoader(
                DataSet(data["pdb_id"].values[:5000],
                        cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
            val_dataloader = DataLoader(
                DataSet(data["pdb_id"].values[5000:7500],
                        cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)
            is_success=True
        except Exception as e:
            print(e)
    def objective(trial: optuna.trial.Trial):

        block_num = trial.suggest_int("block_num", 3, 4, log=True)
        kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
        pool_type = trial.suggest_categorical("pool", ["max", "ave"])
        pool_kernel_size = trial.suggest_int("pool_kernel_size", 2, 2, log=True)

        f_map = [
            trial.suggest_int("channels_{}".format(i), 32, 252, log=True) for i in range(block_num)
        ]

        lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)

        # epoch = trial.suggest_int("epoch", 10, 20, step=10)

        #drop_out = trial.suggest_uniform("drop_out", 0.0, 1.0)

        hyperparameters = dict(block_num=block_num, kernel_size=kernel_size, f_map=f_map, pool_type=pool_type, pool_kernel_size=pool_kernel_size,
                            in_channel=7, out_channel=1 if cfg.model.type == "normal" else 3, lr=lr, gpu_id=gpu_id)

        print(hyperparameters)
        model = UNet(AttributeDict(hyperparameters)).to(device)
        # model = torch.nn.DataParallel(
        #     model) if cfg.training.gpu_num > 1 else model
        trainer = pl.Trainer(max_epochs=30,
                             progress_bar_refresh_rate=100,
                             gpus=[gpu_id],
                            #  gpus=[i for i in range(cfg.training.gpu_num)],
                             logger=False,
                             checkpoint_callback=False,
                             #plugins='ddp_sharded',
                            #  accelerator="ddp",
                             callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss"), EarlyStopping(monitor="val_loss", patience =2)])
        model = WrapperModel(model, loss, lr).to(device)

 #      trainer.logger.log_hyperparams(hyperparameters)

        mlflow.pytorch.autolog(log_models=False)
        experiment = mlflow.get_experiment_by_name(f"{cfg.model.type} 1")
        if experiment == None:
            experiment_id = mlflow.create_experiment(f"{cfg.model.type} 1")
        else:
            experiment_id = experiment.experiment_id
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.set_tags(hyperparameters)
            trainer.fit(model, dataloader, val_dataloader)
 #       print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
        # torch.cuda.empty_cache()

        return trainer.callback_metrics["best_loss"].item()

    #pruner = optuna.pruners.PercentilePruner(60)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    is_success=False
    while not is_success:
        try:
            study = optuna.create_study(direction='minimize', load_if_exists=True, pruner=pruner, storage=f"sqlite:///{cfg.model.type}_1.db", study_name=f"{cfg.model.type}")

            #with parallel_backend("multiprocessing", n_jobs=cfg.training.gpu_num):
            study.optimize(objective, timeout=60*60*8, gc_after_trial=True)

            is_success=True

        except Exception as e:
            print(e)
    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial
    study.trials_dataframe().to_csv("trial_result.csv")

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
import time
# from mpi4py import MPI

if __name__ == "__main__":

    # comm = MPI.COMM_WORLD
    # RANK = comm.Get_rank()
    # rank = comm.Get_rank()
    # GPU_ID=rank
    # time.sleep(rank)
    # main()

    # workers = [Process(target=main) for i in range(OmegaConf.load(os.path.join(hydra.utils.to_absolute_path(""), "params.yaml")).training.gpu_num)]
    # for i, worker in enumerate(workers):
    #     # time.sleep(10)
    #     GPU_ID = i
    #     worker.start()
    # main()
    processes = []
    for rank in range(OmegaConf.load(os.path.join(hydra.utils.to_absolute_path(""), "params.yaml")).training.gpu_num):
        p = torch.multiprocessing.Process(target=main)
        GPU_ID = rank
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

