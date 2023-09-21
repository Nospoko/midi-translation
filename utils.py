import os

import torch
import pretty_midi
import pandas as pd
import torch.nn as nn
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan import MidiPiece
from omegaconf import DictConfig
from fortepyan.audio import render as render_audio

from data.tokenizer import MidiEncoder
from modules.encoderdecoder import subsequent_mask


def vocab_sizes(cfg: DictConfig) -> tuple[int, int]:
    bins = cfg.dataset.quantization

    # +3 is for special tokens we don't really use right now

    src_vocab_size = 3 + 88 * bins.dstart * bins.velocity * bins.duration
    if cfg.target == "velocity":
        tgt_vocab_size = 128 + 3
    elif cfg.target == "dstart":
        tgt_vocab_size = cfg.dstart_bins + 3
    else:
        tgt_vocab_size = None
    return src_vocab_size, tgt_vocab_size


def piece_av_files(piece: MidiPiece, save_base: str) -> dict:
    # fixed by Tomek
    mp3_path = save_base + ".mp3"

    if not os.path.exists(mp3_path):
        render_audio.midi_to_mp3(piece.to_midi(), mp3_path)

    pianoroll_path = save_base + ".png"

    if not os.path.exists(pianoroll_path):
        ff.view.draw_pianoroll_with_velocities(piece)
        plt.tight_layout()
        plt.savefig(pianoroll_path)
        plt.clf()

    midi_path = save_base + ".mid"
    if not os.path.exists(midi_path):
        # Add a silent event to make sure the final notes
        # have time to ring out
        midi = piece.to_midi()
        end_time = midi.get_end_time() + 0.2
        pedal_off = pretty_midi.ControlChange(64, 0, end_time)
        midi.instruments[0].control_changes.append(pedal_off)
        midi.write(midi_path)

    paths = {
        "mp3_path": mp3_path,
        "midi_path": midi_path,
        "pianoroll_path": pianoroll_path,
    }
    return paths


def euclidean_distance(out: torch.Tensor, tgt: torch.Tensor):
    labels = out.argmax(1).to(float)
    # euclidean distance
    return torch.dist(labels, tgt.to(float), p=2)


def calculate_average_distance(out: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    labels = out.argmax(1).to(float)
    # average distance between label and target
    return torch.dist(labels, tgt.to(float), p=1) / len(labels)


def learning_rate_schedule(step: int, model_size: int, factor: float, warmup: int) -> float:
    # we have to default the step to 1 for LambdaLR function
    # to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def generate_sequence(
    src_tokens: torch.Tensor,
    tgt_encoder: MidiEncoder,
    pad_idx: int,
    model: nn.Module,
    sequence_size: int,
    device: str = "cpu",
) -> pd.DataFrame:
    src_mask = (src_tokens != pad_idx).unsqueeze(-2)

    sequence = greedy_decode(
        model=model,
        src=src_tokens,
        src_mask=src_mask,
        max_len=sequence_size,
        start_symbol=0,
        device=device,
    )

    out_sequence = tgt_encoder.untokenize(sequence)

    return out_sequence


def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    device: str = "cpu",
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
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    device: str = "cpu",
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
