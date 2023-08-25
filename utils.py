import os

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
