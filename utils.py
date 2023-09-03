import os

import torch
import fortepyan as ff
import matplotlib.pyplot as plt
from fortepyan import MidiPiece
from fortepyan.audio import render as render_audio


def piece_av_files(piece: MidiPiece) -> dict:
    # stolen from Tomek
    midi_file = os.path.basename(piece.source["midi_filename"])
    mp3_path = midi_file.replace(".midi", ".mp3")
    mp3_path = os.path.join("tmp", mp3_path)
    if not os.path.exists(mp3_path):
        render_audio.midi_to_mp3(piece.to_midi(), mp3_path)

    pianoroll_path = midi_file.replace(".midi", ".png")
    pianoroll_path = os.path.join("tmp", pianoroll_path)
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


def distance(out: torch.Tensor, tgt: torch.Tensor):
    labels = out.argmax(1).to(float)
    # euclidean distance
    return torch.dist(labels, tgt.to(float), p=2)


def rate(step: int, model_size: int, factor: float, warmup: int) -> float:
    # we have to default the step to 1 for LambdaLR function
    # to avoid zero raising to negative power.
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))
