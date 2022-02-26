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
from sklearn.model_selection import KFold
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
class WrapperModel(pl.LightningModule):
    def __init__(self, model, loss, lr, model_filename):
        super(WrapperModel, self).__init__()
        self.model = model
        self.loss = loss
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        self.lr = lr
        self.model_filename=model_filename

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, z, y = batch
        p = self(x)
        loss = self.loss(p, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        p = self.forward(x)
        val_loss = self.loss(p, y)
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
        pass
        torch.save(self.model.state_dict(), f"/gs/hs0/tga-science/murata/models/{self.model_filename}.pth")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

@hydra.main(config_name="params.yaml")
def main(cfg: DictConfig) -> None:
    model_type=cfg.model.type
    if model_type == "normal":
        file_name="normal_1.db"
        db_name="normal"
    if model_type == "multi":
        file_name="multi.db"
        db_name="multi"
    gpu_id=0
    pl.seed_everything(0)
    # device = torch.device("cuda:{}".format(gpu_id)) if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    data = pd.read_csv(os.path.join(
        hydra.utils.to_absolute_path(""), cfg.dataset.train_path))
    loss = Loss()
    pdb_id_header = "pdb_id"
    # test_used = ["3bkl", "2oi0"]
    # data = data[~data[pdb_id_header].isin(test_used)]
    test_used = pd.read_csv(os.path.join(
                hydra.utils.to_absolute_path(""), cfg.dataset.test_path))
    data = data[~data[pdb_id_header].isin(test_used)]

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for ith, (train_index, test_index) in enumerate(kf.split(data)):
        if cfg.cross_validation.ith != ith:
            continue
        train_data = data[pdb_id_header].values[train_index]
        val_data = train_data[:int(len(train_index)/5)]
        train_data = train_data[int(len(train_index)/5):]
        print(len(train_data), len(val_data))
        test_data = data[pdb_id_header].values[test_index]
        test_dataloader = DataLoader(
            DataSet(test_data,
                    cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=1, shuffle=False)

        # seed=random.sample(range(11000), k=5000)
        train_dataloader = DataLoader(
                DataSet(train_data,
                        cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, True), batch_size=cfg.training.batch_size, num_workers=4)
        
        val_dataloader = DataLoader(
                DataSet(val_data,
                        cfg.preprocess.cell_size, cfg.preprocess.grid_size, True, False), batch_size=cfg.training.batch_size)
        

        loaded_study = optuna.load_study(study_name=db_name, storage=f"sqlite:///{file_name}")
        best_params = loaded_study.best_params

        hyperparameters = dict(block_num=best_params["block_num"], kernel_size=best_params["kernel_size"],
        f_map=[best_params[f"channels_{i}"] for i in range(best_params["block_num"])],
        pool_type=best_params["pool"], pool_kernel_size=best_params["pool_kernel_size"],
                                latent_dim=0, in_channel=7, out_channel=1 if model_type == "normal" else 3, lr=best_params["lr"], drop_out=0, gpu_id=None)

        # model = UNet(AttributeDict(hyperparameters)).to(device)
        # # model = torch.nn.DataParallel(
        # #     model) if cfg.training.gpu_num > 1 else model
        # trainer = pl.Trainer(max_epochs=30,
        #                         progress_bar_refresh_rate=20,
        #                         gpus=cfg.training.gpu_num,
        #                         accelerator="ddp",
        #                         logger=False,
        #                         checkpoint_callback=ModelCheckpoint(
        #                             dirpath="/gs/hs0/tga-science/murata/models/",
        #                             filename=f"best_{model_type}_{ith}",
        #                             save_top_k=1,
        #                             verbose=False,
        #                             monitor="epoch_val_loss",
        #                             mode="min"
        #                             ),
        #                         callbacks=[EarlyStopping(monitor="epoch_val_loss", patience =2)])
                                            
        # experiment = mlflow.get_experiment_by_name(f"five-cross-validation {model_type} 1")
        # if experiment == None:
        #     experiment_id = mlflow.create_experiment(f"five-cross-validation {model_type} 1")
        # else:
        #     experiment_id = experiment.experiment_id
        # model = WrapperModel(model, loss, best_params["lr"], f"{model_type}_{ith}").to(device)
        # mlflow.pytorch.autolog(log_models=False)
        # with mlflow.start_run(experiment_id=experiment_id) as run:
        #     mlflow.set_tags(hyperparameters)
        #     trainer.fit(model, train_dataloader, val_dataloader)
            
        # torch.save(model.model.state_dict(), f"/gs/hs0/tga-science/murata/models/best_{model_type}_{ith}.pth")
        model = UNet(AttributeDict(hyperparameters))
        model = WrapperModel(model, loss, best_params["lr"], f"{model_type}_{ith}")
        model.load_state_dict(torch.load(f"/gs/hs0/tga-science/murata/models/best_{model_type}_{ith}-v1.ckpt")["state_dict"], strict=False)
        torch.save(model.model.state_dict(), f"/gs/hs0/tga-science/murata/models/best_{model_type}_{ith}.pth")
        model = model.to(device)
        model.eval()
        with open(f"{model_type}_1/5cv-{ith}.csv", "w") as file_object:
            file_object.write("pdbid, loss_c, loss_o, loss_n \n")
            for i, (protein, ligand) in enumerate(test_dataloader):
                pdbid=test_data[i]
                input_data = protein.view(1, 7, cfg.preprocess.grid_size, cfg.preprocess.grid_size, cfg.preprocess.grid_size)
                input_data = input_data.to(device)
                output = model(input_data)
                loss_values = [loss(output[0,i], ligand.to(device)[0,i]).cpu().detach().numpy() for i in range(3)]
                output = output.cpu().detach().numpy()
                
                file_object.write(f"{pdbid}, {loss_values[0]}, {loss_values[1]}, {loss_values[2]} \n")

import time

if __name__ == "__main__":

    main()