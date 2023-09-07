import os
import json
import pickle
import hashlib

import torch
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan import MidiPiece
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig
from fortepyan.audio import render as render_audio

from data.dataset import BinsToVelocityDataset


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


def learning_rate_schedule(step: int, model_size: int, factor: float, warmup: int) -> float:
    # we have to default the step to 1 for LambdaLR function
    # to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def load_cached_dataset(cfg: DictConfig, split="test") -> BinsToVelocityDataset:
    n_dstart_bins, n_duration_bins, n_velocity_bins = cfg.bins.split(" ")
    n_dstart_bins, n_duration_bins, n_velocity_bins = int(n_dstart_bins), int(n_duration_bins), int(n_velocity_bins)

    config_hash = hashlib.sha256()
    config_string = json.dumps(OmegaConf.to_container(cfg)) + split
    config_hash.update(config_string.encode())
    config_hash = config_hash.hexdigest()
    cache_dir = "tmp/datasets"
    print(f"Preparing dataset: {config_hash}")

    dataset_cache_file = f"{config_hash}.pkl"
    dataset_cache_path = os.path.join(cache_dir, dataset_cache_file)

    if os.path.exists(dataset_cache_path):
        file = open(dataset_cache_path, "rb")
        dataset = pickle.load(file)
    else:
        file = open(dataset_cache_path, "wb")
        hf_dataset = load_dataset(cfg.dataset_name, split=split)
        dataset = BinsToVelocityDataset(
            dataset=hf_dataset,
            n_dstart_bins=n_dstart_bins,
            n_velocity_bins=n_velocity_bins,
            n_duration_bins=n_duration_bins,
            sequence_len=cfg.sequence_size,
        )
        pickle.dump(dataset, file)
    file.close()
    return dataset
