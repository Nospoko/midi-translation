import hydra
from omegaconf import OmegaConf, DictConfig

import wandb
from pipeline.dstart import main as dstart_pipeline
from pipeline.velocity import main as velocity_pipeline


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    initialize_wandb(cfg)
    if cfg.target == "velocity":
        velocity_pipeline.main(cfg)
    elif cfg.targer == "dstart":
        dstart_pipeline.main(cfg)


if __name__ == "__main__":
    main()
