import os

import numpy as np
import streamlit as st
from fortepyan import MidiPiece
from datasets import load_dataset

from utils import piece_av_files
from data.quantizer import MidiQuantizer


def tokenization_review_dashboard():
    st.markdown("### Quantization settings")

    n_dstart_bins = st.number_input(label="n_dstart_bins", value=3)
    n_duration_bins = st.number_input(label="n_duration_bins", value=3)
    n_velocity_bins = st.number_input(label="n_velocity_bins", value=3)

    quantizer = MidiQuantizer(
        n_dstart_bins=n_dstart_bins,
        n_duration_bins=n_duration_bins,
        n_velocity_bins=n_velocity_bins,
    )

    split = "train"
    dataset_name = "roszcz/maestro-v1-sustain"
    dataset = load_dataset(dataset_name, split=split)

    cols = st.columns(2)
    with cols[0]:
        st.markdown("### Unchanged")
    with cols[1]:
        st.markdown("### Quantized")

    np.random.seed(137)
    n_samples = 5
    idxs = np.random.randint(len(dataset), size=n_samples)
    for idx in idxs:
        piece = MidiPiece.from_huggingface(dataset[int(idx)])
        quantized_piece = quantizer.quantize_piece(piece)

        av_dir = "tmp/dashboard/common"
        bins = f"{n_dstart_bins}-{n_duration_bins}-{n_velocity_bins}"
        save_name = f"{dataset_name}-{split}-{idx}".replace("/", "_")

        save_base_gt = os.path.join(av_dir, f"{save_name}")
        gt_paths = piece_av_files(piece, save_base=save_base_gt)

        save_base_quantized = os.path.join(av_dir, f"{save_name}-{bins}")
        quantized_piece_paths = piece_av_files(quantized_piece, save_base=save_base_quantized)

        st.json(piece.source)
        cols = st.columns(2)
        with cols[0]:
            st.image(gt_paths["pianoroll_path"])
            st.audio(gt_paths["mp3_path"])

        with cols[1]:
            st.image(quantized_piece_paths["pianoroll_path"])
            st.audio(quantized_piece_paths["mp3_path"])


if __name__ == "__main__":
    tokenization_review_dashboard()
