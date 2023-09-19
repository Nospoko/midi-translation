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
from data.batch import Batch
from model import make_model
from modules.label_smoothing import LabelSmoothing
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset
from utils import vocab_sizes, learning_rate_schedule, calculate_average_distance


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    train_dataset, val_dataset = load_datasets(dataset_name=cfg.dataset_name, dataset_cfg=cfg.dataset)
    train_model(train_dataset, val_dataset, cfg)
    print(cfg.run_name)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
):
    path = f"models/{cfg.run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cfg": OmegaConf.to_object(cfg),
        },
        path,
    )
    print("Saved!", cfg.run_name)


def load_datasets(
    dataset_name: str,
    dataset_cfg: DictConfig,
) -> tuple[MyTokenizedMidiDataset, MyTokenizedMidiDataset]:
    train_dataset = load_cache_dataset(dataset_name=dataset_name, dataset_cfg=dataset_cfg, split="train")
    val_dataset = load_cache_dataset(dataset_name=dataset_name, dataset_cfg=dataset_cfg, split="validation")

    return train_dataset, val_dataset


def initialize_wandb(cfg: DictConfig):
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def train_model(
    train_dataset: MyTokenizedMidiDataset,
    val_dataset: MyTokenizedMidiDataset,
    cfg: DictConfig,
) -> nn.Module:
    # Get the index for padding token
    pad_idx = train_dataset.src_encoder.token_to_id["<blank>"]
    src_vocab_size, tgt_vocab_size = vocab_sizes(cfg)

    # define model parameters and create the model
    model = make_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        n=cfg.model.n,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        h=cfg.model.h,
        dropout=cfg.model.dropout,
    )
    model.to(cfg.device)

    # Set LabelSmoothing as a criterion for loss calculation
    criterion = LabelSmoothing(
        size=tgt_vocab_size,
        padding_idx=pad_idx,
        smoothing=cfg.train.label_smoothing,
    )
    criterion.to(cfg.device)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=8,
    )

    # Define optimizer and learning learning_rate_schedule lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.base_lr, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: learning_rate_schedule(step, cfg.model.d_model, factor=1, warmup=cfg.warmup),
    )
    initialize_wandb(cfg)
    best_test_loss = float("inf")
    for epoch in range(cfg.train.num_epochs):
        model.train()
        print(f"Epoch {epoch}", flush=True)

        # Train model for one epoch
        t_loss, t_dist = train_epoch(
            dataloader=train_dataloader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accum_iter=cfg.train.accum_iter,
            log_frequency=cfg.log_frequency,
            pad_idx=pad_idx,
            device=cfg.device,
        )

        print(f"Epoch {epoch} Validation", flush=True)
        with torch.no_grad():
            model.eval()
            # Evaluate the model on validation set
            v_loss, v_dist = val_epoch(
                dataloader=val_dataloader,
                model=model,
                criterion=criterion,
                pad_idx=pad_idx,
                device=cfg.device,
            )

        if v_loss <= best_test_loss:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                cfg=cfg,
            )
            best_test_loss = v_loss

        # Log validation and training losses
        wandb.log(
            {
                "val/loss_epoch": v_loss,
                "val/dist_epoch": v_dist,
                "train/loss_epoch": t_loss,
                "train/dist_epoch": t_dist,
                "epoch": epoch,
            }
        )
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
    device: str = "cpu",
) -> tuple[float, float]:
    start = time.time()
    total_loss = 0
    total_dist = 0
    tokens = 0
    n_accum = 0
    it = 0

    # create progress bar
    steps = len(dataloader)
    progress_bar = tqdm(dataloader, total=steps)
    for batch in progress_bar:
        src = batch["source_token_ids"].to(device)
        tgt = batch["target_token_ids"].to(device)
        batch = Batch(src=src, tgt=tgt, pad_idx=pad_idx)

        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(batch.tgt_y, "b n -> (b n)")

        loss = criterion(out, target) / batch.ntokens
        loss.backward()

        dist = calculate_average_distance(out, target)

        # Update the model parameters and optimizer gradients every `accum_iter` iterations
        if it % accum_iter == 0 or it == steps - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
        it += 1

        # Update learning learning_rate_schedule lr_scheduler
        lr_scheduler.step()

        # Update loss and token counts
        loss_item = loss.item()
        total_loss += loss.item()
        total_dist += dist
        tokens += batch.ntokens

        # log metrics every log_frequency steps
        if it % log_frequency == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            tok_rate = tokens / elapsed
            progress_bar.set_description(
                f"Step: {it:6d}/{steps} | acc_step: {n_accum:3d} | loss: {loss_item:6.2f} | dist: {dist:6.2f}"
                + f"| tps: {tok_rate:7.1f} | lr: {lr:6.1e}"
            )

            # log the loss each to Weights and Biases
            wandb.log({"train/loss_step": loss.item(), "train/dist_step": dist})

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), total_dist / len(dataloader)


@torch.no_grad()
def val_epoch(
    dataloader: Iterable,
    model: nn.Module,
    criterion: Callable,
    pad_idx: int = 2,
    device: str = "cpu",
) -> tuple[float, float]:
    total_tokens = 0
    total_loss = 0
    tokens = 0
    total_dist = 0

    for batch in tqdm(dataloader):
        src = batch["source_token_ids"].to(device)
        tgt = batch["target_token_ids"].to(device)
        batch = Batch(src=src, tgt=tgt, pad_idx=pad_idx)

        encoded_decoded = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        out = model.generator(encoded_decoded)

        out_rearranged = einops.rearrange(out, "b n d -> (b n) d")
        target = einops.rearrange(batch.tgt_y, "b n -> (b n)")
        loss = criterion(out_rearranged, target) / batch.ntokens

        total_loss += loss.item()
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        total_dist += calculate_average_distance(out_rearranged, target)

    # Return average loss over all tokens and updated train state
    return total_loss / len(dataloader), total_dist / len(dataloader)


if __name__ == "__main__":
    main()
