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


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient(
    ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def train(trainer: pl.Trainer, dataloader: DataLoader, model: pl.LightningModule, tags: dict, experiment_id: int) -> None:
    mlflow.pytorch.autolog()
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(tags)
        trainer.fit(model, dataloader)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss, val_data=None):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.val_data = DataLoader(
            val_data, batch_size=30, num_workers=8)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        p, mean, var = self(x)
        loss = self.loss(p, y, mean, var)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def training_epoch_end(self, _):
        val_loss = 0
        for d in self.val_data:
            x, y = d
            p, mean, var = self.model(x)
            val_loss += self.loss(p, y, mean, var).item()
        self.log("val_loss", val_loss, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


@hydra.main(config_name="params")
def main(cfg: DictConfig) -> None:
    # pl.seed_everything(0)
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    tags = {"voxel_size": cfg.preprocess.grid_size,
            "cell_size": cfg.preprocess.cell_size}
    dataloader = DataLoader(
        DataSet(data["pdb_id"].values[:10000],
                cfg.preprocess.cell_size, cfg.preprocess.grid_size), batch_size=cfg.training.batch_size, num_workers=8)
    val_data = DataSet(data["pdb_id"].values[10000:11000],
                       cfg.preprocess.cell_size, cfg.preprocess.grid_size)

    trainer = pl.Trainer(max_epochs=cfg.training.epoch,
                         progress_bar_refresh_rate=20, gpus=cfg.training.gpu_num)
    m = AutoEncoder(cfg.preprocess.grid_size)
    loss = VAELoss()
    model = WrapperModel(m, loss, val_data=val_data)
    train(trainer, dataloader, model, tags, cfg.model.experiment_id)


if __name__ == "__main__":
    main()
