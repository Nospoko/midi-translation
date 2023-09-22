import glob

import hydra
import torch
from omegaconf import OmegaConf, DictConfig

from model import make_model
from utils import vocab_sizes
from data.dataset import load_cache_dataset
from pipeline.dstart import evaluate as dstart_evaluation
from pipeline.velocity import evaluate as velocity_evaluation


def load_model_checkpoint(cfg: DictConfig) -> dict:
    model_path = None
    for file in glob.glob("checkpoints/*/*.pt"):
        if cfg.run_name in file:
            model_path = file
    if model_path is None:
        raise FileNotFoundError()
    return torch.load(model_path, map_location=cfg.device)


@hydra.main(version_base=None, config_path="../configs", config_name="eval_conf")
def main(cfg: DictConfig):
    checkpoint = load_model_checkpoint(cfg)
    train_cfg = OmegaConf.create(checkpoint["cfg"])

    src_vocab_size, tgt_vocab_size = vocab_sizes(train_cfg)
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        n=train_cfg.model.n,
        d_model=train_cfg.model.d_model,
        d_ff=train_cfg.model.d_ff,
        h=train_cfg.model.h,
        dropout=train_cfg.model.dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(cfg.device)

    translation_dataset = load_cache_dataset(train_cfg.dataset, dataset_name=cfg.dataset_name, split=cfg.split)
    if train_cfg.target == "velocity":
        velocity_evaluation.main(
            cfg=train_cfg,
            model=model,
            translation_dataset=translation_dataset,
            device=cfg.device,
        )
    elif train_cfg.target == "dstart":
        dstart_evaluation.main(
            cfg=train_cfg,
            model=model,
            translation_dataset=translation_dataset,
            device=cfg.device,
        )


if __name__ == "__main__":
    main()
