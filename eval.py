from train import val_epoch
from model import make_model
from modules.label_smoothing import LabelSmoothing
from torch.utils.data import DataLoader
from data.dataset import TokenizedMidiDataset
import torch
import os
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="eval_conf")
def main(cfg):
    checkpoint = load_checkpoint(
        run_name=cfg.run_name,
        epoch=cfg.model_epoch,
        device=cfg.device,
    )
    train_cfg = OmegaConf.create(checkpoint["cfg"])
    val_data = TokenizedMidiDataset(
        split='test',
        n_dstart_bins=3,
        n_velocity_bins=3,
        n_duration_bins=3,
        sequence_len=train_cfg.sequence_size,
        device=cfg.device,
    )

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

    if cfg.tasks.val_epoch:

        pad_idx = val_data.tgt_vocab.index("<blank>")

        criterion = LabelSmoothing(
            size=output_size,
            padding_idx=pad_idx,
            smoothing=train_cfg.train.label_smoothing,
        )
        criterion.to(cfg.device)

        print("Evaluating model ...")
        loss = val_epoch(
            dataloader=dataloader,
            model=model,
            criterion=criterion,
        )
        print(f"Model loss:   {loss}")

    # if cfg.tasks.translation:



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
