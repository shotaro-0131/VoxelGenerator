import hydra
from omegaconf import DictConfig, omegaconf
from models.charge_model_vae import *


def train(trainer: Trainer) -> None:
    trainer.run()
    
@hydra.main(config_name="params")
def main(cfg: DictConfig) -> None:
    trainer = get_trainer(cfg.dataset.train_path)
    trainer.batch_size = cfg.training.batch_size
    trainer.epoch = cfg.training.epoch
    train(trainer)

if __name__ == "__main__":
    main()