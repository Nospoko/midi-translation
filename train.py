import hydra
from datasets import concatenate_datasets
from omegaconf import OmegaConf, DictConfig

import wandb
from data.dataset import load_cache_dataset
from data.augmentation import augment_dataset
from pipeline.dstart import main as dstart_pipeline
from pipeline.velocity import main as velocity_pipeline


def load_train_dataset(cfg: DictConfig):
    datasets = []
    for name in cfg.dataset_name.split("+"):
        dataset = load_cache_dataset(
            dataset_cfg=cfg.dataset,
            dataset_name=name,
            split="train",
        )
        datasets.append(dataset)
    train_dataset = concatenate_datasets(datasets)

    return train_dataset


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    # apply DRY: common translation datasets creation before pipelines
    train_translation_dataset = load_train_dataset(cfg)
    val_translation_dataset = load_cache_dataset(
        dataset_cfg=cfg.dataset,
        dataset_name="roszcz/maestro-v1-sustain",
        split="test+validation",
    )

    if cfg.augmentation.repetitions > 0:
        train_translation_dataset = augment_dataset(
            dataset=train_translation_dataset,
            dataset_cfg=cfg.dataset,
            augmentation_cfg=cfg.augmentation,
        )

    initialize_wandb(cfg)

    if cfg.target == "velocity":
        velocity_pipeline.main(cfg, train_translation_dataset, val_translation_dataset)
    elif cfg.target == "dstart":
        dstart_pipeline.main(cfg, train_translation_dataset, val_translation_dataset)


if __name__ == "__main__":
    main()
