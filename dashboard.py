import pandas as pd
import torch
import numpy as np
import os

from data.dataset import TokenizedMidiDataset
import streamlit as st
from fortepyan import MidiPiece
from utils import piece_av_files


def main():
    dataset = TokenizedMidiDataset(split='validation')
    n_samples = 10

    cols = st.columns(2)
    indexes = torch.randint(0, len(dataset), [n_samples])
    for idx in indexes:

        record = dataset.unprocessed_records[idx]

        processed_record = dataset.processed_records[idx]
        processed_df = pd.DataFrame(processed_record)

        filename = processed_df.pop('midi_filename')[0]
        print(filename)

        notes = pd.DataFrame(record)
        quantized_notes = dataset.quantizer.apply_quantization(processed_df)

        start_time = np.min(notes["start"])

        notes["start"] -= start_time
        notes["end"] -= start_time
        start_time = np.min(processed_df["start"])
        processed_df["start"] -= start_time
        processed_df["end"] -= start_time

        piece = MidiPiece(notes)
        name = filename.split("/")[0] + "/" + str(idx) + '/real'
        piece.source["midi_filename"] = name + os.path.basename(filename)
        paths = piece_av_files(piece)

        quantized_piece = MidiPiece(quantized_notes)
        name = filename.split("/")[0] + "/" + str(idx) + '/quantized'
        quantized_piece.source["midi_filename"] = name + os.path.basename(filename)
        quantized_piece_paths = piece_av_files(quantized_piece)

        with cols[0]:
            st.image(paths["pianoroll_path"])
            st.audio(paths["mp3_path"])
            st.table(piece.source)

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])
            st.table(quantized_piece.source)


if __name__ == '__main__':
    main()