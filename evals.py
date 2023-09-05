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
    val_data = load_cached_dataset(train_cfg.dataset)

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

    if cfg.tasks.translation:
        print("Checking model outputs ...")
        translations = make_examples(
            dataset=val_data,
            model=model,
            n_examples=cfg.n_examples,
            random=cfg.random,
        )

        for translation in translations:
            print("Source (Input)        : " + " ".join(translation["src"]))
            print("Target (Ground Truth) : " + " ".join(translation["tgt"]))
            print("Model Output               : " + " ".join(translation["out"]))


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


def make_examples(
    dataset: BinsToVelocityDataset,
    model: nn.Module,
    start_index: int = 0,
    n_examples: int = 5,
    eos_string: str = "</s>",
    random: bool = False,
):
    results = []
    pad_idx = dataset.tgt_vocab.index("<blank>")
    dataloader = DataLoader(dataset, shuffle=random)
    idx = -1
    for b in dataloader:
        batch = Batch(b[0], b[1], pad_idx)
        for it in range(len(batch)):
            idx += 1
            # I want to be able to get samples from any index from the database,
            # I also want to use examples without overlap
            if idx < start_index or idx % 2 == 1:
                continue

            record = batch[it]
            src_tokens = [dataset.src_vocab[x] for x in record.src if x != pad_idx]
            tgt_tokens = [dataset.tgt_vocab[x] for x in record.tgt if x != pad_idx]

            decoded_record = greedy_decode(
                model=model,
                src=record.src,
                src_mask=record.src_mask,
                max_len=dataset.sequence_len,
                start_symbol=0,
            )

            model_txt = [dataset.tgt_vocab[x] for x in decoded_record if x != pad_idx]
            result = {
                "src": src_tokens,
                "tgt": tgt_tokens,
                "out": model_txt,
            }
            results.append(result)

            if len(results) == n_examples:
                return results

    return results


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
) -> torch.Tensor:
    # Pretend to be batches
    src = src.unsqueeze(0)
    src_mask = src_mask.unsqueeze(0)

    memory = model.encode(src, src_mask)
    # Create a tensor and put start symbol inside
    sentence = torch.Tensor([[start_symbol]]).type_as(src.data)
    for _ in range(max_len):
        sub_mask = subsequent_mask(sentence.size(1)).type_as(src.data)
        out = model.decode(memory, src_mask, sentence, sub_mask)

        prob = model.generator(out[:, -1])
        _, next_word = prob.max(dim=1)
        next_word = next_word.data[0]

        sentence = torch.cat([sentence, torch.Tensor([[next_word]]).type_as(src.data)], dim=1)

    # Don't pretend to be a batch
    return sentence[0]


if __name__ == "__main__":
    main()
