import time
from typing import Callable, Iterable

import hydra
import torch
import einops
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from torch.optim.lr_scheduler import LambdaLR

import wandb
from utils import rate
from data.batch import Batch
from model import make_model
from data.dataset import TokenizedMidiDataset
from modules.label_smoothing import LabelSmoothing


@hydra.main(version_base=None, config_path="config", config_name="conf")
def main(cfg: DictConfig):
    n_dstart_bins, n_duration_bins, n_velocity_bins = cfg.bins.split(" ")
    n_dstart_bins, n_duration_bins, n_velocity_bins = int(n_dstart_bins), int(n_duration_bins), int(n_velocity_bins)
    bins = "-".join(cfg.bins)
    train_data = TokenizedMidiDataset(
        split="train",
        n_dstart_bins=n_dstart_bins,
        n_velocity_bins=n_velocity_bins,
        n_duration_bins=n_duration_bins,
        sequence_len=cfg.sequence_size,
        device=cfg.device,
    )
    val_data = TokenizedMidiDataset(
        split="validation",
        n_dstart_bins=n_dstart_bins,
        n_velocity_bins=n_velocity_bins,
        n_duration_bins=n_duration_bins,
        sequence_len=cfg.sequence_size,
        device=cfg.device,
    )

    model = train_model(train_data, val_data, cfg)
    path = f"models/{bins}-{cfg.file_prefix}-{cfg.run_name}-final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": OmegaConf.to_container(cfg),
            "input_size": len(train_data.src_vocab),
            "output_size": len(train_data.tgt_vocab),
        },
        path,
    )


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def train_model(
    train_data: TokenizedMidiDataset,
    val_data: TokenizedMidiDataset,
    cfg: DictConfig,
) -> nn.Module:
    # Get the index for padding token
    pad_idx = train_data.tgt_vocab.index("<blank>")
    vocab_src_size = len(train_data.src_vocab)
    vocab_tgt_size = len(train_data.tgt_vocab)

    # define model parameters and create the model
    model = make_model(
        input_size=vocab_src_size,
        output_size=vocab_tgt_size,
        n=cfg.model.n,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        h=cfg.model.h,
        dropout=cfg.model.dropout,
    )
    model.to(cfg.device)

    # Set LabelSmoothing as a criterion for loss calculation
    criterion = LabelSmoothing(
        size=vocab_tgt_size,
        padding_idx=pad_idx,
        smoothing=cfg.train.label_smoothing,
    )
    criterion.to(cfg.device)

    train_dataloader = DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=cfg.train.batch_size, shuffle=True)

    # Define optimizer and learning rate lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, cfg.model.d_model, factor=1, warmup=cfg.warmup),
    )
    initialize_wandb(cfg)

    for epoch in range(cfg.train.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accum_iter=cfg.train.accum_iter,
            log_frequency=cfg.log_frequency,
            pad_idx=pad_idx,
        )

        bins = "-".join(cfg.bins)
        # Save checkpoint after each epoch
        file_path = f"models/{bins}-{cfg.file_prefix}-{cfg.run_name}-{epoch}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "cfg": OmegaConf.to_container(cfg),
                "input_size": len(train_data.src_vocab),
                "output_size": len(train_data.tgt_vocab),
            },
            file_path,
        )

        print(f"Epoch {epoch} Validation", flush=True)
        with torch.no_grad():
            model.eval()
            # Evaluate the model on validation set
            v_loss = val_epoch(dataloader=val_dataloader, model=model, criterion=criterion, pad_idx=pad_idx)
            print(float(v_loss))

        # Log validation and training losses
        wandb.log({"val/loss_epoch": v_loss, "train/loss_epoch": t_loss})
    return model


def train_epoch(
    dataloader: Iterable,
    model: nn.Module,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LambdaLR,
    accum_iter: int = 1,
    log_frequency: int = 10,
    pad_idx: int = 2,
) -> float:
    start = time.time()
    total_loss = 0
    tokens = 0
    n_accum = 0
    it = 0

    # create progress bar
    steps = len(dataloader)
    pbar = tqdm(dataloader, total=steps)

    for b in pbar:
        batch = Batch(b[0], b[1], pad=pad_idx)
        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out, target) / batch.ntokens
        loss.backward()

        # Update the model parameters and optimizer gradients every `accum_iter` iterations
        if it % accum_iter == 0 or it == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
        it += 1

        # Update learning rate lr_scheduler
        lr_scheduler.step()

        # Update loss and token counts
        loss_item = loss.item()
        total_loss += loss.item()
        tokens += batch.ntokens

        # log metrics every log_frequency steps
        if it % log_frequency == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_rate = tokens / elapsed
            pbar.set_description(
                f"Step: {it:6d}/{steps} | acc_step: {n_accum:3d} | Loss: {loss_item:6.2f}"
                + f"| tps: {tok_rate:7.1f} | LR: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            wandb.log({"train/loss_step": loss.item()})

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader)


@torch.no_grad()
def val_epoch(
    dataloader: Iterable,
    model: nn.Module,
    criterion: Callable,
    pad_idx: int = 2,
) -> float:
    total_tokens = 0
    total_loss = 0
    tokens = 0

    for b in tqdm(dataloader):
        batch = Batch(b[0], b[1], pad=pad_idx)
        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        tgt_rearranged = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out_rearranged, tgt_rearranged) / batch.ntokens

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader)


if __name__ == "__main__":
    main()