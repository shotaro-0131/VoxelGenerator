import hydra
from omegaconf import DictConfig, omegaconf
from models.charge_model_vae import *
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import pytorch_lightning as pl
from torch.utils.data import DataLoader


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient(
    ).list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


def train(trainer: pl.Trainer, dataloader: DataLoader, model: pl.LightningModule) -> None:
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        trainer.fit(model, dataloader)
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))


class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        p, mean, var = self(x)
        loss = self.loss(p, y, mean, var)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


@hydra.main(config_name="params")
def main(cfg: DictConfig) -> None:
    # pl.seed_everything(0)
    
    dataloader = DataLoader(
        get_dataset(hydra.utils.to_absolute_path("")+cfg.dataset.train_path), batch_size=cfg.training.batch_size, num_workers=8)
    trainer = pl.Trainer(max_epochs=cfg.training.epoch,
                         progress_bar_refresh_rate=20, gpus=1)
    m = ConcatVAEModel()
    loss = VAELoss()
    model = WrapperModel(m, loss)
    train(trainer, dataloader, model)


if __name__ == "__main__":
    main()
