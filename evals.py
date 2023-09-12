import os

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from train import val_epoch
from data.batch import Batch
from model import make_model
from utils import load_cached_dataset
from data.dataset import BinsToVelocityDataset
from modules.encoderdecoder import subsequent_mask
from modules.label_smoothing import LabelSmoothing


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg):
    checkpoint = load_checkpoint(
        run_name=cfg.run_name,
        epoch=cfg.model_epoch,
        device=cfg.device,
    )
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    if cfg.dataset.dataset_name is None:
        val_data = load_cached_dataset(train_cfg.dataset, split=cfg.dataset_split)
    else:
        cfg.dataset.bins = train_cfg.dataset.bins
        cfg.dataset.sequence_size = train_cfg.dataset.sequence_size
        val_data = load_cached_dataset(cfg.dataset, split=cfg.dataset_split)

    dataloader = DataLoader(val_data, batch_size=train_cfg.train.batch_size)

    input_size = len(val_data.src_vocab)
    output_size = len(val_data.tgt_vocab)

    model = make_model(
        input_size=input_size,
        output_size=output_size,
        n=train_cfg.model.n,
        d_model=train_cfg.model.d_model,
        d_ff=train_cfg.model.d_ff,
        dropout=train_cfg.model.dropout,
    )
    model.to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    pad_idx = val_data.tgt_vocab.index("<blank>")

    criterion = LabelSmoothing(
        size=output_size,
        padding_idx=pad_idx,
        smoothing=train_cfg.train.label_smoothing,
    )
    criterion.to(cfg.device)

    print("Evaluating model ...")
    loss, dist = val_epoch(
        dataloader=dataloader,
        model=model,
        criterion=criterion,
        device=cfg.device,
    )
    print(f"Model loss:   {loss} | Average distance:    {dist}")


def load_checkpoint(run_name: str, epoch: str = "final", device: str = "cpu"):
    # find path with desired run
    path = None
    for file in os.listdir("models"):
        if file.find(f"{run_name}-{epoch}") > 0:
            path = file
            break

    if path is None:
        print("no run with this id found")
        return None

    path = "models/" + path
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


if __name__ == "__main__":
    main()
