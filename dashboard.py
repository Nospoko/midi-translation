import os

import torch
import numpy as np
import pandas as pd
import streamlit as st
from fortepyan import MidiPiece

from utils import piece_av_files
from data.dataset import TokenizedMidiDataset


def main():
    dataset = TokenizedMidiDataset(split="validation")
    n_samples = 10

    cols = st.columns(2)
    with cols[0]:
        st.header("Source sample")
    with cols[1]:
        st.header("Quantized sample")

    indexes = torch.randint(0, len(dataset), [n_samples])
    for idx in indexes:
        piece, quantized_piece = prepare_midi_pieces(
            unprocessed=dataset.unprocessed_records[idx],
            processed=dataset.processed_records[idx],
            idx=idx,
            dataset=dataset,
        )

        paths = piece_av_files(piece)
        quantized_piece_paths = piece_av_files(quantized_piece)

        with cols[0]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(piece.source)

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])
            st.table(quantized_piece.source)


def prepare_midi_pieces(
    unprocessed: dict,
    processed: dict,
    idx: int,
    dataset: TokenizedMidiDataset,
) -> tuple[MidiPiece, MidiPiece]:
    processed_df = pd.DataFrame(processed)

    filename = processed_df.pop("midi_filename")[0]
    print(filename)

    notes = pd.DataFrame(unprocessed)
    quantized_notes = dataset.quantizer.apply_quantization(processed_df)

    start_time = np.min(notes["start"])

    notes["start"] -= start_time
    notes["end"] -= start_time
    start_time = np.min(processed_df["start"])
    processed_df["start"] -= start_time
    processed_df["end"] -= start_time

    piece = MidiPiece(notes)
    name = filename.split("/")[0] + "/" + str(idx) + "/real"
    piece.source["midi_filename"] = name + os.path.basename(filename)

    quantized_piece = MidiPiece(quantized_notes)
    name = filename.split("/")[0] + "/" + str(idx) + "/quantized"
    quantized_piece.source["midi_filename"] = name + os.path.basename(filename)
    return piece, quantized_piece


if __name__ == "__main__":
    main()
