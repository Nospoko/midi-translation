import os

import torch
import torch.nn as nn
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan import MidiPiece
from fortepyan.audio import render as render_audio
# BinsToVelocity and BinsToDstart datasets must be imported to use eval(cfg.dataset_class) in load_cached_dataset
from data.dataset import TokenizedMidiDataset
from modules.encoderdecoder import subsequent_mask


def piece_av_files(piece: MidiPiece) -> dict:
    # stolen from Tomek
    midi_file = piece.source["midi_filename"]
    mp3_path = midi_file.replace(".midi", ".mp3").replace(".mid", ".mp3")

    if not os.path.exists(mp3_path):
        render_audio.midi_to_mp3(piece.to_midi(), mp3_path)

    pianoroll_path = midi_file.replace(".midi", ".png").replace(".mid", ".png")

    if not os.path.exists(pianoroll_path):
        ff.view.draw_pianoroll_with_velocities(piece)
        plt.tight_layout()
        plt.savefig(pianoroll_path)
        plt.clf()

    paths = {
        "mp3_path": mp3_path,
        "pianoroll_path": pianoroll_path,
    }
    return paths


def euclidean_distance(out: torch.Tensor, tgt: torch.Tensor):
    labels = out.argmax(1).to(float)
    # euclidean distance
    return torch.dist(labels, tgt.to(float), p=2)


def avg_distance(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    labels = out.argmax(1).to(float)
    # average distance between label and target
    return torch.dist(labels, tgt.to(float), p=1) / len(labels)


def learning_rate_schedule(step: int, model_size: int, factor: float, warmup: int) -> float:
    # we have to default the step to 1 for LambdaLR function
    # to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def predict_sample(record, dataset: TokenizedMidiDataset, model, cfg, train_cfg):
    pad_idx = dataset.tgt_vocab.index("<blank>")
    src_mask = (record[0] != pad_idx).unsqueeze(-2)

    sequence = greedy_decode(
        model=model,
        src=record[0],
        src_mask=src_mask,
        max_len=train_cfg.dataset.sequence_size,
        start_symbol=0,
        device=cfg.device,
    )

    out_tokens = [dataset.tgt_vocab[x] for x in sequence if x != pad_idx]

    return out_tokens


def greedy_decode(
    model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int, device: str = "cpu"
) -> torch.Tensor:
    dev = torch.device(device)
    # Pretend to be batches
    src = src.unsqueeze(0).to(dev)
    src_mask = src_mask.unsqueeze(0).to(dev)

    memory = model.encode(src, src_mask)
    # Create a tensor and put start symbol inside
    sentence = torch.Tensor([[start_symbol]]).type_as(src.data).to(dev)
    for _ in range(max_len):
        sub_mask = subsequent_mask(sentence.size(1)).type_as(src.data).to(dev)
        out = model.decode(memory, src_mask, sentence, sub_mask)

        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=1)
        next_word = next_word.data[0]

        sentence = torch.cat([sentence, torch.Tensor([[next_word]]).type_as(src.data).to(dev)], dim=1)

    # Don't pretend to be a batch
    return sentence[0]


def decode_and_output(
    model: nn.Module, src: torch.Tensor, src_mask: torch.Tensor, max_len: int, start_symbol: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device)
    # Pretend to be batches
    src = src.unsqueeze(0).to(dev)
    src_mask = src_mask.unsqueeze(0).to(dev)

    memory = model.encode(src, src_mask)
    # Create a tensor and put start symbol inside
    sentence = torch.Tensor([[start_symbol]]).type_as(src.data).to(dev)
    probabilities = torch.Tensor([]).to(dev)
    for _ in range(max_len):
        sub_mask = subsequent_mask(sentence.size(1)).type_as(src.data).to(dev)
        out = model.decode(memory, src_mask, sentence, sub_mask)

        prob = model.generator(out[:, -1])
        next_word = prob.argmax(dim=1)
        next_word = next_word.data[0]

        sentence = torch.cat([sentence, torch.Tensor([[next_word]]).type_as(src.data).to(dev)], dim=1)
        probabilities = torch.cat([probabilities, prob], dim=0)
    # Don't pretend to be a batch
    return sentence[0], probabilities
